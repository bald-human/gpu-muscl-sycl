#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <chrono>

namespace py = pybind11;

class SyclQueuemanager {
public:
    // parameter initilization
    sycl::queue q_; 
    bool initilized_q = false;

    std::chrono::milliseconds total_time_;
    size_t total_steps_;

    size_t dim0_;
    size_t dim1_;
    size_t dim2_;
    
    size_t N_;
    size_t nv_;
    size_t ndim_;

    double cs_;
    double gamma_;
    double Cdt_;

    size_t iD_ = 0;
    size_t iE_ = 1;
    size_t ivx_;
    size_t ivy_;
    size_t ivz_;

    bool isothermal_ = false;

    // nullpointer initilization
    double* vars_sycl_ = nullptr;
    double* ptr_vars = nullptr;
    std::size_t vars_size_ = 0;
    bool initilized_vars = false;

    double* prim_sycl_    = nullptr;
    double* dprim_sycl_   = nullptr;
    double* predict_sycl_ = nullptr;
    double* facel_sycl_   = nullptr;
    double* facer_sycl_   = nullptr;
    double* flux_sycl_    = nullptr;
    double* invD_sycl_    = nullptr;
    double* ptr_ds_       = nullptr;
    double* ds_sycl_      = nullptr;
    double* d_max_        = nullptr;
    int* iw_order_1_      = nullptr;
    int* iw_order_2_      = nullptr;
    int* iF_order_1_      = nullptr;
    int* iF_order_2_      = nullptr;
    double* min_dt_       = nullptr;
    bool arrays_allocated_ = false;

public:
    // make and get sycl::queue
    sycl::queue& get_queue() {
        if (!initilized_q){
            try {
                q_ = sycl::queue(sycl::gpu_selector_v);
                std::cout << "SYCL Queue initialized on: " 
                        << q_.get_device().get_info<sycl::info::device::name>() 
                        << std::endl;
            } catch (const sycl::exception& e) {
                        std::cerr << "Failed to create GPU queue: " << e.what() << std::endl;
                        std::cout << "Falling back to CPU queue..." << std::endl;
                        q_ = sycl::queue(sycl::cpu_selector_v);
                    }
        initilized_q = true;
        }
        return q_;
    }
    // initilize and save all parameters used in the simulation
    void init_params(py::list n, size_t nv, size_t ndim, bool isothermal, py::slice iM, double cs, double gamma, double Cdt){
        dim0_ = n[0].cast<size_t>();
        dim1_ = n[1].cast<size_t>();
        dim2_ = n[2].cast<size_t>();

        N_ = dim0_*dim1_*dim2_;
        nv_ = nv;
        ndim_ = ndim;
        
        // initilize variables for the python slice 
        ssize_t start, stop, step, slice_length;    
        // make iM slice into usable variables
        if (!iM.compute(nv, &start, &stop, &step, &slice_length))   
        throw std::runtime_error("Invalid slice");
    
        ivx_ = start;         // indicies for Momentum 
        ivy_ = start + 1;
        ivz_ = start + 2;  

        cs_ = cs;
        gamma_ = gamma;
        Cdt_= Cdt; 
    }   
    // allocate vars on the GPU  
    void init_vars_from_host(py::array_t<double> vars){
        // allocate once on device
        if (!initilized_vars){
            auto& q = get_queue();
            py::buffer_info vars_buf = vars.request();
            ptr_vars = static_cast<double*>(vars_buf.ptr);
            vars_size_ = vars_buf.size;
            vars_sycl_ = sycl::malloc_device<double>(vars_size_, q);

            initilized_vars = true;
        }
        // copy host data into existing device allocation
        get_queue().memcpy(vars_sycl_, ptr_vars, vars_size_ * sizeof(double)).wait();
    }
    // copy vars array from CPU to GPU
    void cpy_vars_from_device(){
        get_queue().memcpy(ptr_vars, vars_sycl_, vars_size_ * sizeof(double)).wait();
        sycl::free(vars_sycl_, get_queue());
    }
    double* get_vars_sycl(){
        if (!initilized_vars){ throw std::runtime_error("vars_sycl not initialized");}
        return vars_sycl_;
    }
    // allocate all neccessary arrays for the simulation ahead of time
    void malloc_arrays(py::array_t<double> ds_) {
        if (arrays_allocated_) return;
        auto& q = get_queue(); 
       
        int IF_ORDER_1[5] = {0, 1, 4, 2, 3};
        int IF_ORDER_2[5] = {0, 1, 3, 4, 2};
        int IW_ORDER_1[5] = {0, 1, 3, 4, 2};
        int IW_ORDER_2[5] = {0, 1, 4, 2, 3};

        prim_sycl_ =    sycl::malloc_device<double>(N_*nv_, q);
        dprim_sycl_ =   sycl::malloc_device<double>(N_*nv_*ndim_, q);
        predict_sycl_ = sycl::malloc_device<double>(N_*nv_, q);
        facel_sycl_ =   sycl::malloc_device<double>(N_*nv_, q);
        facer_sycl_ =   sycl::malloc_device<double>(N_*nv_, q);
        flux_sycl_    = sycl::malloc_device<double>(N_*nv_, q);
        invD_sycl_    = sycl::malloc_device<double>(N_, q);
        ds_sycl_   =    sycl::malloc_device<double>(3, q);
        d_max_ =        sycl::malloc_device<double>(3, q);
        iw_order_1_ =   sycl::malloc_device<int>(5, q);
        iw_order_2_ =   sycl::malloc_device<int>(5, q);
        iF_order_1_ =   sycl::malloc_device<int>(5, q);
        iF_order_2_ =   sycl::malloc_device<int>(5, q);
        min_dt_     =   sycl::malloc_device<double>(1, q);
        
        // host pointer for ds_
        py::buffer_info ds_buf = ds_.request();
        if (ds_buf.size < 3) {
            throw std::runtime_error("ds_ must have at least 3 elements");
        }
        double* ds_host = static_cast<double*>(ds_buf.ptr);
        ptr_ds_ = static_cast<double*>(ds_buf.ptr);

        q.memcpy(ds_sycl_,    ds_host,        3 * sizeof(double));
        q.fill(d_max_, 0.0, 3);
        q.fill(min_dt_, 0.0, 1);
        q.memcpy(iw_order_1_, IW_ORDER_1, 5 * sizeof(int));
        q.memcpy(iw_order_2_, IW_ORDER_2, 5 * sizeof(int));
        q.memcpy(iF_order_1_, IF_ORDER_1, 5 * sizeof(int));
        q.memcpy(iF_order_2_, IF_ORDER_2, 5 * sizeof(int)).wait();

        arrays_allocated_ = true;
    } 
    // free all allocated arrays on the GPU after simulation
    void free_arrays() {
        auto& q = get_queue();
        sycl::free(prim_sycl_, q);
        sycl::free(dprim_sycl_, q);
        sycl::free(predict_sycl_, q);
        sycl::free(facel_sycl_, q);
        sycl::free(facer_sycl_, q);
        sycl::free(flux_sycl_, q);
        sycl::free(invD_sycl_, q);
        sycl::free(ds_sycl_, q);
        sycl::free(d_max_, q);
        sycl::free(iw_order_1_, q);
        sycl::free(iw_order_2_, q);
        sycl::free(iF_order_1_, q);
        sycl::free(iF_order_2_, q); 
        sycl::free(min_dt_, q);
    }
    // getter functions for arrays  
    double* get_prim_sycl()    { return prim_sycl_; }
    double* get_dprim_sycl()   { return dprim_sycl_; }
    double* get_facel_sycl()   { return facel_sycl_; }
    double* get_facer_sycl()   { return facer_sycl_; }
    double* get_predict_sycl() { return predict_sycl_; }
    double* get_flux_sycl()    { return flux_sycl_ ; }
    double* get_invD_sycl()    { return invD_sycl_ ; }
    double* get_ds_sycl()      { return ds_sycl_; }
    double* get_ptr_ds()       { return ptr_ds_; } 
    double* get_d_max()        { return d_max_ ; }
    int*    get_iw_order_1()   { return iw_order_1_; }
    int*    get_iw_order_2()   { return iw_order_2_; }
    int*    get_iF_order_1()   { return iF_order_1_; }
    int*    get_iF_order_2()   { return iF_order_2_; }
    double* get_min_dt()           { return min_dt_; }

    std::string get_device_info() {
        auto dev = get_queue().get_device();
        return "Device: " + dev.get_info<sycl::info::device::name>();
    }    
};

// template function for pointers from pybind arrays
template <typename T> T* get_array_ptr(py::array_t< T > a){

    py::buffer_info a_buf = a.request();
    T* a_ptr = static_cast< T* >(a_buf.ptr);
    return a_ptr;
}

void moncen_sycl(sycl::queue& q, double* prim_sycl, double* dprim_sycl, double* ds_sycl, size_t nv, size_t dim0, size_t dim1, size_t dim2){
    size_t N = dim0*dim1*dim2;
    
    // 2D range for variables and spatial dimensions
    q.parallel_for(sycl::range<2>(nv, N), [=](sycl::id<2> idx) {
        size_t iv = idx[0];
        size_t it = idx[1];

        // convert to 3D spatial coordinates
        size_t iz = it % dim2;
        size_t iy = (it/ dim2) % dim1;
        size_t ix = it / (dim1 * dim2);

        // current cell index in prim
        size_t current_idx = iv*N + ix * dim1 * dim2 + iy * dim2 + iz;

        // X-direction slope
        size_t ixp = iv*N + ((ix + 1) % dim0) * (dim1 * dim2) + iy * dim2 + iz;
        size_t ixm = iv*N + ((ix + dim0 - 1) % dim0) * (dim1 * dim2) + iy * dim2 + iz;
        double f_xm = prim_sycl[ixm];
        double lsx = 0.5 * (prim_sycl[current_idx] - f_xm);
        double f_xp = prim_sycl[ixp];
        double rsx = 0.5 * (f_xp - prim_sycl[current_idx]);
        double wx = lsx * rsx;
        double slope_x = (wx > 0) ? (2.0 * wx / ((lsx + rsx)*ds_sycl[0])) : 0.0;
        dprim_sycl[iv*N*3 + 0*N + ix * dim1 * dim2 + iy * dim2 + iz] = slope_x;

        // Y-direction slope
        size_t iyp = (iy + 1) % dim1;
        size_t iym = (iy + dim1 - 1) % dim1;
        double f_ym = prim_sycl[iv*N + ix * (dim1 * dim2) + iym * dim2 + iz];
        double lsy = 0.5 * (prim_sycl[current_idx] - f_ym);
        double f_yp = prim_sycl[iv*N + ix * (dim1 * dim2) + iyp * dim2 + iz];
        double rsy = 0.5 * (f_yp - prim_sycl[current_idx]);
        double wy = lsy * rsy;
        double slope_y = (wy > 0) ? (2.0 * wy / ((lsy + rsy)*ds_sycl[1])) : 0.0;
        dprim_sycl[iv*N*3 + 1*N + ix * dim1 * dim2 + iy * dim2 + iz] = slope_y;

        // Z-direction slope
        size_t izp = (iz + 1) % dim2;
        size_t izm = (iz + dim2 - 1) % dim2;
        double f_zm = prim_sycl[iv*N + ix * (dim1 * dim2) + iy * dim2 + izm];
        double lsz = 0.5 * (prim_sycl[current_idx] - f_zm);
        double f_zp = prim_sycl[iv*N + ix * (dim1 * dim2) + iy * dim2 + izp];
        double rsz = 0.5 * (f_zp - prim_sycl[current_idx]);
        double wz = lsz * rsz;
        double slope_z = (wz > 0) ? (2.0 * wz / ((lsz + rsz)*ds_sycl[2])) : 0.0;
        dprim_sycl[iv*N*3 + 2*N + ix * dim1 * dim2 + iy * dim2 + iz] = slope_z; 
    }).wait();
}

double* HLL_combine(sycl::queue& q, double* flux, double* ql, double* qr, int N, int nv, int iD, int iE, int ivx, int ivy, int ivz, bool isothermal, double cs, double gamma){
    if (!isothermal) {
        double gamma_factor = 1.0/(gamma-1.0);
        q.parallel_for(N, [=](sycl::id<1> it){
        
            double cl_iD = ql[iD*N+it];
            double Etot_l = ql[ivx*N+it]*ql[ivx*N+it] + ql[ivy*N+it]*ql[ivy*N+it] + ql[ivz*N+it]*ql[ivz*N+it];
            Etot_l *= 0.5*ql[iD*N+it];
            Etot_l += gamma_factor * ql[iE*N+it];
            double cl_iE = Etot_l;
            double cl_ivx = ql[iD*N+it]*ql[ivx*N+it];  // momentum
            double cl_ivy = ql[iD*N+it]*ql[ivy*N+it];
            double cl_ivz = ql[iD*N+it]*ql[ivz*N+it];

            double cr_iD = qr[iD*N+it];
            double Etot_r = qr[ivx*N+it]*qr[ivx*N+it] + qr[ivy*N+it]*qr[ivy*N+it] + qr[ivz*N+it]*qr[ivz*N+it];
            Etot_r *= 0.5*qr[iD*N+it];
            Etot_r += gamma_factor * qr[iE*N+it];
            double cr_iE = Etot_r;
            double cr_ivx = qr[iD*N+it]*qr[ivx*N+it];  // momentum
            double cr_ivy = qr[iD*N+it]*qr[ivy*N+it];
            double cr_ivz = qr[iD*N+it]*qr[ivz*N+it];

            double Fl_iD = cl_ivx;         // Normal velocity v = q[ivx]
            double Fl_ivx = cl_ivx*ql[ivx*N+it];           // Velocity part of momentum flux F_v[i] = D v[i] v_norm
            double Fl_ivy = cl_ivy*ql[ivx*N+it];
            double Fl_ivz = cl_ivz*ql[ivx*N+it];
            double Fl_iE = (cl_iE+ql[iE*N+it])*ql[ivx*N+it];        // Energy flux = (E + P) v
            Fl_ivx += ql[iE*N+it];  

            double Fr_iD = cr_ivx;         // Normal velocity v = q[ivx]
            double Fr_ivx = cr_ivx*qr[ivx*N+it];           // Velocity part of momentum flux F_v[i] = D v[i] v_norm
            double Fr_ivy = cr_ivy*qr[ivx*N+it];
            double Fr_ivz = cr_ivz*qr[ivx*N+it];
            double Fr_iE = (cr_iE+qr[iE*N+it])*qr[ivx*N+it];        // Energy flux = (E + P) v
            Fr_ivx += qr[iE*N+it];  

            double c2l = gamma * ql[iE*N+it]/ql[iD*N+it];
            double c2r = gamma * qr[iE*N+it]/qr[iD*N+it];
            double c_max = sycl::sqrt(sycl::max(c2l, c2r));            
            
            double SL = sycl::min(sycl::min(ql[ivx*N+it], qr[ivx*N+it]) - c_max, 0.0);
            double SR = sycl::max(sycl::max(ql[ivx*N+it], qr[ivx*N+it]) + c_max, 0.0);
            double iSRL = 1.0/(SR - SL);
            double SRL = SR*SL;

            flux[iD*N+it] = (SR*Fl_iD - SL*Fr_iD + SRL*(cr_iD - cl_iD))*iSRL;      
            flux[iE*N+it] = (SR*Fl_iE - SL*Fr_iE + SRL*(cr_iE - cl_iE))*iSRL;      
            flux[ivx*N+it] = (SR*Fl_ivx - SL*Fr_ivx + SRL*(cr_ivx - cl_ivx))*iSRL;      
            flux[ivy*N+it] = (SR*Fl_ivy - SL*Fr_ivy + SRL*(cr_ivy - cl_ivy))*iSRL;      
            flux[ivz*N+it] = (SR*Fl_ivz - SL*Fr_ivz + SRL*(cr_ivz - cl_ivz))*iSRL;      

        }).wait();
    } else {
        double cs_factor = cs*cs; 
        q.parallel_for(N, [=](sycl::id<1> it){
            double cl_iD = ql[iD*N+it];
            double cl_ivx = ql[iD*N+it]*ql[ivx*N+it];  // momentum
            double cl_ivy = ql[iD*N+it]*ql[ivy*N+it];
            double cl_ivz = ql[iD*N+it]*ql[ivz*N+it];

            double cr_iD = qr[iD*N+it];
            double cr_ivx = qr[iD*N+it]*qr[ivx*N+it];  // momentum
            double cr_ivy = qr[iD*N+it]*qr[ivy*N+it];
            double cr_ivz = qr[iD*N+it]*qr[ivz*N+it];
            
            double Fl_iD = cl_ivx;         // Normal velocity v = q[ivx]
            double Fl_ivx = cl_ivx*ql[ivx*N+it];           // Velocity part of momentum flux F_v[i] = D v[i] v_norm
            double Fl_ivy = cl_ivy*ql[ivx*N+it];
            double Fl_ivz = cl_ivz*ql[ivx*N+it];
            Fl_ivx += cs_factor*cl_iD;  

            double Fr_iD = cr_ivx;         // Normal velocity v = q[ivx]
            double Fr_ivx = cr_ivx*qr[ivx*N+it];           // Velocity part of momentum flux F_v[i] = D v[i] v_norm
            double Fr_ivy = cr_ivy*qr[ivx*N+it];
            double Fr_ivz = cr_ivz*qr[ivx*N+it];
            Fr_ivx += cs_factor*cr_iD;
            
            double c_max = cs;

            double SL = sycl::min(sycl::min(ql[ivx*N+it], qr[ivx*N+it]) - c_max, 0.0);
            double SR = sycl::max(sycl::max(ql[ivx*N+it], qr[ivx*N+it]) + c_max, 0.0);
            double iSRL = 1.0/(SR - SL);
            double SRL = SR*SL;

            flux[iD*N+it] = (SR*Fl_iD - SL*Fr_iD + SRL*(cr_iD - cl_iD))*iSRL;      
            flux[ivx*N+it] = (SR*Fl_ivx - SL*Fr_ivx + SRL*(cr_ivx - cl_ivx))*iSRL;      
            flux[ivy*N+it] = (SR*Fl_ivy - SL*Fr_ivy + SRL*(cr_ivy - cl_ivy))*iSRL;      
            flux[ivz*N+it] = (SR*Fl_ivz - SL*Fr_ivz + SRL*(cr_ivz - cl_ivz))*iSRL;      

        }).wait();    
    }
    return flux;
    }

void calc_faces_0(sycl::queue& q, double* facel_sycl, double* facer_sycl, double* prim_sycl, double* dprim_sycl, double* invD_sycl, double* ds_sycl, size_t N, ssize_t iD, ssize_t iE, ssize_t ivx, ssize_t ivy, ssize_t ivz, size_t ndim, size_t dim0, size_t dim1, size_t dim2, bool isothermal, double cs, double gamma, double dt) {    
    if (!isothermal){ 
        q.parallel_for(N, [=](sycl::id<1> it){
            // Convert to 3D spatial coordinates
            size_t iz = it % dim2;
            size_t iy = (it / dim2) % dim1;
            size_t ix = it / (dim1 * dim2);
            size_t ixm = ((ix + dim0 - 1) % dim0) * (dim1 * dim2) + iy * dim2 + iz;
            double hds = 0.5 * ds_sycl[0];      // x-direction 
            // ========== LOAD FOR CELL (it) ==========
            double prim_ivx_it = prim_sycl[ivx*N+it];
            double prim_ivy_it = prim_sycl[ivy*N+it];
            double prim_ivz_it = prim_sycl[ivz*N+it];
            double invD_it = invD_sycl[it];

            double dprim_iD_0_it = dprim_sycl[iD*N*ndim+0*N+it];
            double dprim_iD_1_it = dprim_sycl[iD*N*ndim+1*N+it];
            double dprim_iD_2_it = dprim_sycl[iD*N*ndim+2*N+it];
            
            double dprim_iE_0_it = dprim_sycl[iE*N*ndim+0*N+it];
            double dprim_iE_1_it = dprim_sycl[iE*N*ndim+1*N+it];
            double dprim_iE_2_it = dprim_sycl[iE*N*ndim+2*N+it];
            
            double dprim_ivx_0_it = dprim_sycl[ivx*N*ndim+0*N+it];
            double dprim_ivx_1_it = dprim_sycl[ivx*N*ndim+1*N+it];
            double dprim_ivx_2_it = dprim_sycl[ivx*N*ndim+2*N+it];
            
            double dprim_ivy_0_it = dprim_sycl[ivy*N*ndim+0*N+it];
            double dprim_ivy_1_it = dprim_sycl[ivy*N*ndim+1*N+it];
            double dprim_ivy_2_it = dprim_sycl[ivy*N*ndim+2*N+it];
            
            double dprim_ivz_0_it = dprim_sycl[ivz*N*ndim+0*N+it];
            double dprim_ivz_1_it = dprim_sycl[ivz*N*ndim+1*N+it];
            double dprim_ivz_2_it = dprim_sycl[ivz*N*ndim+2*N+it];
            
            double div_v_trace_it = dprim_ivx_0_it + dprim_ivy_1_it + dprim_ivz_2_it;
           
            // ========== COMPUTE PREDICT FOR CELL (it) ==========

            double sum_v_dD = prim_ivx_it*dprim_iD_0_it + prim_ivy_it*dprim_iD_1_it + prim_ivz_it*dprim_iD_2_it;
            double predict_iD_it = prim_sycl[iD*N+it] - 0.5*dt*(sum_v_dD + prim_sycl[iD*N+it]*div_v_trace_it);
            
            double sum_v_dP = prim_ivx_it*dprim_iE_0_it + prim_ivy_it*dprim_iE_1_it + prim_ivz_it*dprim_iE_2_it;
            double predict_iE_it = prim_sycl[iE*N+it] - 0.5*dt*(sum_v_dP + gamma*prim_sycl[iE*N+it]*div_v_trace_it);

            double sum_v_dvx = prim_ivx_it*dprim_ivx_0_it + prim_ivy_it*dprim_ivx_1_it + prim_ivz_it*dprim_ivx_2_it;
            double predict_ivx_it = prim_ivx_it - 0.5*dt*(sum_v_dvx + invD_it*dprim_iE_0_it);

            double sum_v_dvy = prim_ivx_it*dprim_ivy_0_it + prim_ivy_it*dprim_ivy_1_it + prim_ivz_it*dprim_ivy_2_it;
            double predict_ivy_it = prim_ivy_it - 0.5*dt*(sum_v_dvy + invD_it*dprim_iE_1_it);
            
            double sum_v_dvz = prim_ivx_it*dprim_ivz_0_it + prim_ivy_it*dprim_ivz_1_it + prim_ivz_it*dprim_ivz_2_it;
            double predict_ivz_it = prim_ivz_it - 0.5*dt*(sum_v_dvz + invD_it*dprim_iE_2_it);
            
            // ========== COMPUTE FACES ==========
            facel_sycl[iD*N+it] = predict_iD_it + hds * dprim_iD_0_it;
            facer_sycl[iD*N+ixm] = predict_iD_it - hds * dprim_iD_0_it; 

            facel_sycl[iE*N+it] = predict_iE_it + hds * dprim_iE_0_it;
            facer_sycl[iE*N+ixm] = predict_iE_it - hds * dprim_iE_0_it;

            facel_sycl[ivx*N+it] = predict_ivx_it + hds * dprim_ivx_0_it;
            facer_sycl[ivx*N+ixm] = predict_ivx_it - hds * dprim_ivx_0_it;

            facel_sycl[ivy*N+it] = predict_ivy_it + hds * dprim_ivy_0_it;
            facer_sycl[ivy*N+ixm] = predict_ivy_it - hds * dprim_ivy_0_it;

            facel_sycl[ivz*N+it] = predict_ivz_it + hds * dprim_ivz_0_it;
            facer_sycl[ivz*N+ixm] = predict_ivz_it - hds * dprim_ivz_0_it;
        }).wait();
    } else {
        double factor = cs*cs;
        q.parallel_for(N, [=](sycl::id<1> it){
            // Convert to 3D spatial coordinates
            size_t iz = it % dim2;
            size_t iy = (it / dim2) % dim1;
            size_t ix = it / (dim1 * dim2);
            size_t ixm = ((ix + dim0 - 1) % dim0) * (dim1 * dim2) + iy * dim2 + iz;
            double hds = 0.5 * ds_sycl[0];      // x-direction 
            // ========== LOAD FOR CELL (it) ==========
            double prim_ivx_it = prim_sycl[ivx*N+it];
            double prim_ivy_it = prim_sycl[ivy*N+it];
            double prim_ivz_it = prim_sycl[ivz*N+it];
            double invD_it = invD_sycl[it];

            double dprim_iD_0_it = dprim_sycl[iD*N*ndim+0*N+it];
            double dprim_iD_1_it = dprim_sycl[iD*N*ndim+1*N+it];
            double dprim_iD_2_it = dprim_sycl[iD*N*ndim+2*N+it];
            
            double dprim_ivx_0_it = dprim_sycl[ivx*N*ndim+0*N+it];
            double dprim_ivx_1_it = dprim_sycl[ivx*N*ndim+1*N+it];
            double dprim_ivx_2_it = dprim_sycl[ivx*N*ndim+2*N+it];
            
            double dprim_ivy_0_it = dprim_sycl[ivy*N*ndim+0*N+it];
            double dprim_ivy_1_it = dprim_sycl[ivy*N*ndim+1*N+it];
            double dprim_ivy_2_it = dprim_sycl[ivy*N*ndim+2*N+it];
            
            double dprim_ivz_0_it = dprim_sycl[ivz*N*ndim+0*N+it];
            double dprim_ivz_1_it = dprim_sycl[ivz*N*ndim+1*N+it];
            double dprim_ivz_2_it = dprim_sycl[ivz*N*ndim+2*N+it];
            
            double div_v_trace_it = dprim_ivx_0_it + dprim_ivy_1_it + dprim_ivz_2_it;
            
            // ========== COMPUTE PREDICT FOR CELL (it) ==========

            double sum_v_dD = prim_ivx_it*dprim_iD_0_it + prim_ivy_it*dprim_iD_1_it + prim_ivz_it*dprim_iD_2_it;
            double predict_iD_it = prim_sycl[iD*N+it] - 0.5*dt*(sum_v_dD + prim_sycl[iD*N+it]*div_v_trace_it);

            double sum_v_dvx = prim_ivx_it*dprim_ivx_0_it + prim_ivy_it*dprim_ivx_1_it + prim_ivz_it*dprim_ivx_2_it;
            double predict_ivx_it = prim_ivx_it - 0.5*dt*(sum_v_dvx + factor*invD_it*dprim_iD_0_it);

            double sum_v_dvy = prim_ivx_it*dprim_ivy_0_it + prim_ivy_it*dprim_ivy_1_it + prim_ivz_it*dprim_ivy_2_it;
            double predict_ivy_it = prim_ivy_it - 0.5*dt*(sum_v_dvy + factor*invD_it*dprim_iD_1_it);
            
            double sum_v_dvz = prim_ivx_it*dprim_ivz_0_it + prim_ivy_it*dprim_ivz_1_it + prim_ivz_it*dprim_ivz_2_it;
            double predict_ivz_it = prim_ivz_it - 0.5*dt*(sum_v_dvz + factor*invD_it*dprim_iD_2_it);
            
            // ========== COMPUTE FACES ==========

            facel_sycl[iD*N+it] = predict_iD_it + hds * dprim_iD_0_it;
            facer_sycl[iD*N+ixm] = predict_iD_it - hds * dprim_iD_0_it; 

            facel_sycl[ivx*N+it] = predict_ivx_it + hds * dprim_ivx_0_it;
            facer_sycl[ivx*N+ixm] = predict_ivx_it - hds * dprim_ivx_0_it;

            facel_sycl[ivy*N+it] = predict_ivy_it + hds * dprim_ivy_0_it;
            facer_sycl[ivy*N+ixm] = predict_ivy_it - hds * dprim_ivy_0_it;

            facel_sycl[ivz*N+it] = predict_ivz_it + hds * dprim_ivz_0_it;
            facer_sycl[ivz*N+ixm] = predict_ivz_it - hds * dprim_ivz_0_it;

        }).wait();
    }
}

void calc_faces_1(sycl::queue& q, double* facel_sycl, double* facer_sycl, double* prim_sycl, double* dprim_sycl, double* invD_sycl, double* ds_sycl, size_t N, ssize_t iD, ssize_t iE, ssize_t ivx, ssize_t ivy, ssize_t ivz, size_t ndim, size_t dim0, size_t dim1, size_t dim2, bool isothermal, double cs, double gamma, double dt) {    
    if (!isothermal){ 
        q.parallel_for(N, [=](sycl::id<1> it){

            // Convert to 3D spatial coordinates
            size_t iz = it % dim2;
            size_t iy = (it / dim2) % dim1;
            size_t ix = it / (dim1 * dim2);
            size_t itym = ix * (dim1 * dim2) + ((iy + dim1 - 1) % dim1) * dim2 + iz;
            double hds = 0.5 * ds_sycl[1];      // y-direction 
            // ========== LOAD FOR CELL (it) ==========
            double prim_ivx_it = prim_sycl[ivx*N+it];
            double prim_ivy_it = prim_sycl[ivy*N+it];
            double prim_ivz_it = prim_sycl[ivz*N+it];
            double invD_it = invD_sycl[it];

            double dprim_iD_0_it = dprim_sycl[iD*N*ndim+0*N+it];
            double dprim_iD_1_it = dprim_sycl[iD*N*ndim+1*N+it];
            double dprim_iD_2_it = dprim_sycl[iD*N*ndim+2*N+it];
            
            double dprim_iE_0_it = dprim_sycl[iE*N*ndim+0*N+it];
            double dprim_iE_1_it = dprim_sycl[iE*N*ndim+1*N+it];
            double dprim_iE_2_it = dprim_sycl[iE*N*ndim+2*N+it];
            
            double dprim_ivx_0_it = dprim_sycl[ivx*N*ndim+0*N+it];
            double dprim_ivx_1_it = dprim_sycl[ivx*N*ndim+1*N+it];
            double dprim_ivx_2_it = dprim_sycl[ivx*N*ndim+2*N+it];
            
            double dprim_ivy_0_it = dprim_sycl[ivy*N*ndim+0*N+it];
            double dprim_ivy_1_it = dprim_sycl[ivy*N*ndim+1*N+it];
            double dprim_ivy_2_it = dprim_sycl[ivy*N*ndim+2*N+it];
            
            double dprim_ivz_0_it = dprim_sycl[ivz*N*ndim+0*N+it];
            double dprim_ivz_1_it = dprim_sycl[ivz*N*ndim+1*N+it];
            double dprim_ivz_2_it = dprim_sycl[ivz*N*ndim+2*N+it];
            
            double div_v_trace_it = dprim_ivx_0_it + dprim_ivy_1_it + dprim_ivz_2_it;
           
            // ========== COMPUTE PREDICT FOR CELL (it) ==========
            double sum_v_dD = prim_ivx_it*dprim_iD_0_it + prim_ivy_it*dprim_iD_1_it + prim_ivz_it*dprim_iD_2_it;
            double predict_iD_it = prim_sycl[iD*N+it] - 0.5*dt*(sum_v_dD + prim_sycl[iD*N+it]*div_v_trace_it);
            
            double sum_v_dP = prim_ivx_it*dprim_iE_0_it + prim_ivy_it*dprim_iE_1_it + prim_ivz_it*dprim_iE_2_it;
            double predict_iE_it = prim_sycl[iE*N+it] - 0.5*dt*(sum_v_dP + gamma*prim_sycl[iE*N+it]*div_v_trace_it);

            double sum_v_dvx = prim_ivx_it*dprim_ivx_0_it + prim_ivy_it*dprim_ivx_1_it + prim_ivz_it*dprim_ivx_2_it;
            double predict_ivx_it = prim_ivx_it - 0.5*dt*(sum_v_dvx + invD_it*dprim_iE_0_it);

            double sum_v_dvy = prim_ivx_it*dprim_ivy_0_it + prim_ivy_it*dprim_ivy_1_it + prim_ivz_it*dprim_ivy_2_it;
            double predict_ivy_it = prim_ivy_it - 0.5*dt*(sum_v_dvy + invD_it*dprim_iE_1_it);
            
            double sum_v_dvz = prim_ivx_it*dprim_ivz_0_it + prim_ivy_it*dprim_ivz_1_it + prim_ivz_it*dprim_ivz_2_it;
            double predict_ivz_it = prim_ivz_it - 0.5*dt*(sum_v_dvz + invD_it*dprim_iE_2_it);
            
            // ========== COMPUTE FACES ==========
            facel_sycl[iD*N+it] = predict_iD_it + hds * dprim_iD_1_it;
            facer_sycl[iD*N+itym] = predict_iD_it - hds * dprim_iD_1_it; 

            facel_sycl[iE*N+it] = predict_iE_it + hds * dprim_iE_1_it;
            facer_sycl[iE*N+itym] = predict_iE_it - hds * dprim_iE_1_it;

            facel_sycl[ivx*N+it] = predict_ivy_it + hds * dprim_ivy_1_it;
            facer_sycl[ivx*N+itym] = predict_ivy_it - hds * dprim_ivy_1_it;

            facel_sycl[ivy*N+it] = predict_ivz_it + hds * dprim_ivz_1_it;
            facer_sycl[ivy*N+itym] = predict_ivz_it - hds * dprim_ivz_1_it;

            facel_sycl[ivz*N+it] = predict_ivx_it + hds * dprim_ivx_1_it;
            facer_sycl[ivz*N+itym] = predict_ivx_it - hds * dprim_ivx_1_it;
        }).wait();
    } else {
        double factor = cs*cs;
        q.parallel_for(N, [=](sycl::id<1> it){
        
            // Convert to 3D spatial coordinates
            size_t iz = it % dim2;
            size_t iy = (it / dim2) % dim1;
            size_t ix = it / (dim1 * dim2);
            size_t itym = ix * (dim1 * dim2) + ((iy + dim1 - 1) % dim1) * dim2 + iz;
            double hds = 0.5 * ds_sycl[1];      // y-direction 
            // ========== LOAD FOR CELL (it) ==========
            double prim_ivx_it = prim_sycl[ivx*N+it];
            double prim_ivy_it = prim_sycl[ivy*N+it];
            double prim_ivz_it = prim_sycl[ivz*N+it];
            double invD_it = invD_sycl[it];

            double dprim_iD_0_it = dprim_sycl[iD*N*ndim+0*N+it];
            double dprim_iD_1_it = dprim_sycl[iD*N*ndim+1*N+it];
            double dprim_iD_2_it = dprim_sycl[iD*N*ndim+2*N+it];
            
            double dprim_ivx_0_it = dprim_sycl[ivx*N*ndim+0*N+it];
            double dprim_ivx_1_it = dprim_sycl[ivx*N*ndim+1*N+it];
            double dprim_ivx_2_it = dprim_sycl[ivx*N*ndim+2*N+it];
            
            double dprim_ivy_0_it = dprim_sycl[ivy*N*ndim+0*N+it];
            double dprim_ivy_1_it = dprim_sycl[ivy*N*ndim+1*N+it];
            double dprim_ivy_2_it = dprim_sycl[ivy*N*ndim+2*N+it];
            
            double dprim_ivz_0_it = dprim_sycl[ivz*N*ndim+0*N+it];
            double dprim_ivz_1_it = dprim_sycl[ivz*N*ndim+1*N+it];
            double dprim_ivz_2_it = dprim_sycl[ivz*N*ndim+2*N+it];
            
            double div_v_trace_it = dprim_ivx_0_it + dprim_ivy_1_it + dprim_ivz_2_it;
            
            // ========== COMPUTE PREDICT FOR CELL (it) ==========
            double sum_v_dD = prim_ivx_it*dprim_iD_0_it + prim_ivy_it*dprim_iD_1_it + prim_ivz_it*dprim_iD_2_it;
            double predict_iD_it = prim_sycl[iD*N+it] - 0.5*dt*(sum_v_dD + prim_sycl[iD*N+it]*div_v_trace_it);
            
            double sum_v_dvx = prim_ivx_it*dprim_ivx_0_it + prim_ivy_it*dprim_ivx_1_it + prim_ivz_it*dprim_ivx_2_it;
            double predict_ivx_it = prim_ivx_it - 0.5*dt*(sum_v_dvx + factor*invD_it*dprim_iD_0_it);

            double sum_v_dvy = prim_ivx_it*dprim_ivy_0_it + prim_ivy_it*dprim_ivy_1_it + prim_ivz_it*dprim_ivy_2_it;
            double predict_ivy_it = prim_ivy_it - 0.5*dt*(sum_v_dvy + factor*invD_it*dprim_iD_1_it);
            
            double sum_v_dvz = prim_ivx_it*dprim_ivz_0_it + prim_ivy_it*dprim_ivz_1_it + prim_ivz_it*dprim_ivz_2_it;
            double predict_ivz_it = prim_ivz_it - 0.5*dt*(sum_v_dvz + factor*invD_it*dprim_iD_2_it);
            
            // ========== COMPUTE FACES ==========
            facel_sycl[iD*N+it] = predict_iD_it + hds * dprim_iD_1_it;
            facer_sycl[iD*N+itym] = predict_iD_it - hds * dprim_iD_1_it; 

            facel_sycl[ivx*N+it] = predict_ivy_it + hds * dprim_ivy_1_it;
            facer_sycl[ivx*N+itym] = predict_ivy_it - hds * dprim_ivy_1_it;

            facel_sycl[ivy*N+it] = predict_ivz_it + hds * dprim_ivz_1_it;
            facer_sycl[ivy*N+itym] = predict_ivz_it - hds * dprim_ivz_1_it;

            facel_sycl[ivz*N+it] = predict_ivx_it + hds * dprim_ivx_1_it;
            facer_sycl[ivz*N+itym] = predict_ivx_it - hds * dprim_ivx_1_it;
        }).wait();
    }
}

void calc_faces_2(sycl::queue& q, double* facel_sycl, double* facer_sycl, double* prim_sycl, double* dprim_sycl, double* invD_sycl, double* ds_sycl, size_t N, ssize_t iD, ssize_t iE, ssize_t ivx, ssize_t ivy, ssize_t ivz, size_t ndim, size_t dim0, size_t dim1, size_t dim2, bool isothermal, double cs, double gamma, double dt) {    
    if (!isothermal){ 
        q.parallel_for(N, [=](sycl::id<1> it){
            // Convert to 3D spatial coordinates
            size_t iz = it % dim2;
            size_t iy = (it / dim2) % dim1;
            size_t ix = it / (dim1 * dim2);
            size_t itzm = ix * (dim1 * dim2) + iy * dim2 + (iz + dim2 - 1) % dim2;
            double hds = 0.5 * ds_sycl[2];      // z-direction 
            // ========== LOAD FOR CELL (it) ==========
            double prim_ivx_it = prim_sycl[ivx*N+it];
            double prim_ivy_it = prim_sycl[ivy*N+it];
            double prim_ivz_it = prim_sycl[ivz*N+it];
            double invD_it = invD_sycl[it];

            double dprim_iD_0_it = dprim_sycl[iD*N*ndim+0*N+it];
            double dprim_iD_1_it = dprim_sycl[iD*N*ndim+1*N+it];
            double dprim_iD_2_it = dprim_sycl[iD*N*ndim+2*N+it];
            
            double dprim_iE_0_it = dprim_sycl[iE*N*ndim+0*N+it];
            double dprim_iE_1_it = dprim_sycl[iE*N*ndim+1*N+it];
            double dprim_iE_2_it = dprim_sycl[iE*N*ndim+2*N+it];
            
            double dprim_ivx_0_it = dprim_sycl[ivx*N*ndim+0*N+it];
            double dprim_ivx_1_it = dprim_sycl[ivx*N*ndim+1*N+it];
            double dprim_ivx_2_it = dprim_sycl[ivx*N*ndim+2*N+it];
            
            double dprim_ivy_0_it = dprim_sycl[ivy*N*ndim+0*N+it];
            double dprim_ivy_1_it = dprim_sycl[ivy*N*ndim+1*N+it];
            double dprim_ivy_2_it = dprim_sycl[ivy*N*ndim+2*N+it];
            
            double dprim_ivz_0_it = dprim_sycl[ivz*N*ndim+0*N+it];
            double dprim_ivz_1_it = dprim_sycl[ivz*N*ndim+1*N+it];
            double dprim_ivz_2_it = dprim_sycl[ivz*N*ndim+2*N+it];
            
            double div_v_trace_it = dprim_ivx_0_it + dprim_ivy_1_it + dprim_ivz_2_it;
           
            // ========== COMPUTE PREDICT FOR CELL (it) ==========
            double sum_v_dD = prim_ivx_it*dprim_iD_0_it + prim_ivy_it*dprim_iD_1_it + prim_ivz_it*dprim_iD_2_it;
            double predict_iD_it = prim_sycl[iD*N+it] - 0.5*dt*(sum_v_dD + prim_sycl[iD*N+it]*div_v_trace_it);
            
            double sum_v_dP = prim_ivx_it*dprim_iE_0_it + prim_ivy_it*dprim_iE_1_it + prim_ivz_it*dprim_iE_2_it;
            double predict_iE_it = prim_sycl[iE*N+it] - 0.5*dt*(sum_v_dP + gamma*prim_sycl[iE*N+it]*div_v_trace_it);

            double sum_v_dvx = prim_ivx_it*dprim_ivx_0_it + prim_ivy_it*dprim_ivx_1_it + prim_ivz_it*dprim_ivx_2_it;
            double predict_ivx_it = prim_ivx_it - 0.5*dt*(sum_v_dvx + invD_it*dprim_iE_0_it);

            double sum_v_dvy = prim_ivx_it*dprim_ivy_0_it + prim_ivy_it*dprim_ivy_1_it + prim_ivz_it*dprim_ivy_2_it;
            double predict_ivy_it = prim_ivy_it - 0.5*dt*(sum_v_dvy + invD_it*dprim_iE_1_it);
            
            double sum_v_dvz = prim_ivx_it*dprim_ivz_0_it + prim_ivy_it*dprim_ivz_1_it + prim_ivz_it*dprim_ivz_2_it;
            double predict_ivz_it = prim_ivz_it - 0.5*dt*(sum_v_dvz + invD_it*dprim_iE_2_it);
            
            // ========== COMPUTE FACES ==========
            facel_sycl[iD*N+it] = predict_iD_it + hds * dprim_iD_2_it;
            facer_sycl[iD*N+itzm] = predict_iD_it - hds * dprim_iD_2_it; 

            facel_sycl[iE*N+it] = predict_iE_it + hds * dprim_iE_2_it;
            facer_sycl[iE*N+itzm] = predict_iE_it - hds * dprim_iE_2_it;

            facel_sycl[ivx*N+it] = predict_ivz_it + hds * dprim_ivz_2_it;
            facer_sycl[ivx*N+itzm] = predict_ivz_it - hds * dprim_ivz_2_it;

            facel_sycl[ivy*N+it] = predict_ivx_it + hds * dprim_ivx_2_it;
            facer_sycl[ivy*N+itzm] = predict_ivx_it - hds * dprim_ivx_2_it;

            facel_sycl[ivz*N+it] = predict_ivy_it + hds * dprim_ivy_2_it;
            facer_sycl[ivz*N+itzm] = predict_ivy_it - hds * dprim_ivy_2_it;
        }).wait();
    } else {
        double factor = cs*cs;
        q.parallel_for(N, [=](sycl::id<1> it){
        
            // Convert to 3D spatial coordinates
            size_t iz = it % dim2;
            size_t iy = (it / dim2) % dim1;
            size_t ix = it / (dim1 * dim2);
            size_t itzm = ix * (dim1 * dim2) + iy * dim2 + (iz + dim2 - 1) % dim2;
            double hds = 0.5 * ds_sycl[2];      // z-direction 
            // ========== LOAD FOR CELL (it) ==========
            double prim_ivx_it = prim_sycl[ivx*N+it];
            double prim_ivy_it = prim_sycl[ivy*N+it];
            double prim_ivz_it = prim_sycl[ivz*N+it];
            double invD_it = invD_sycl[it];
            
            double dprim_iD_0_it = dprim_sycl[iD*N*ndim+0*N+it];
            double dprim_iD_1_it = dprim_sycl[iD*N*ndim+1*N+it];
            double dprim_iD_2_it = dprim_sycl[iD*N*ndim+2*N+it];
            
            double dprim_iE_0_it = dprim_sycl[iE*N*ndim+0*N+it];
            double dprim_iE_1_it = dprim_sycl[iE*N*ndim+1*N+it];
            double dprim_iE_2_it = dprim_sycl[iE*N*ndim+2*N+it];
            
            double dprim_ivx_0_it = dprim_sycl[ivx*N*ndim+0*N+it];
            double dprim_ivx_1_it = dprim_sycl[ivx*N*ndim+1*N+it];
            double dprim_ivx_2_it = dprim_sycl[ivx*N*ndim+2*N+it];
            
            double dprim_ivy_0_it = dprim_sycl[ivy*N*ndim+0*N+it];
            double dprim_ivy_1_it = dprim_sycl[ivy*N*ndim+1*N+it];
            double dprim_ivy_2_it = dprim_sycl[ivy*N*ndim+2*N+it];
            
            double dprim_ivz_0_it = dprim_sycl[ivz*N*ndim+0*N+it];
            double dprim_ivz_1_it = dprim_sycl[ivz*N*ndim+1*N+it];
            double dprim_ivz_2_it = dprim_sycl[ivz*N*ndim+2*N+it];
            
            double div_v_trace_it = dprim_ivx_0_it + dprim_ivy_1_it + dprim_ivz_2_it;
            
            // ========== COMPUTE PREDICT FOR CELL (it) ==========
            double sum_v_dD = prim_ivx_it*dprim_iD_0_it + prim_ivy_it*dprim_iD_1_it + prim_ivz_it*dprim_iD_2_it;
            double predict_iD_it = prim_sycl[iD*N+it] - 0.5*dt*(sum_v_dD + prim_sycl[iD*N+it]*div_v_trace_it);
    
            double sum_v_dvx = prim_ivx_it*dprim_ivx_0_it + prim_ivy_it*dprim_ivx_1_it + prim_ivz_it*dprim_ivx_2_it;
            double predict_ivx_it = prim_ivx_it - 0.5*dt*(sum_v_dvx + factor*invD_it*dprim_iD_0_it);

            double sum_v_dvy = prim_ivx_it*dprim_ivy_0_it + prim_ivy_it*dprim_ivy_1_it + prim_ivz_it*dprim_ivy_2_it;
            double predict_ivy_it = prim_ivy_it - 0.5*dt*(sum_v_dvy + factor*invD_it*dprim_iD_1_it);
            
            double sum_v_dvz = prim_ivx_it*dprim_ivz_0_it + prim_ivy_it*dprim_ivz_1_it + prim_ivz_it*dprim_ivz_2_it;
            double predict_ivz_it = prim_ivz_it - 0.5*dt*(sum_v_dvz + factor*invD_it*dprim_iD_2_it);
            
            // ========== COMPUTE FACES ==========
            facel_sycl[iD*N+it] = predict_iD_it + hds * dprim_iD_2_it;
            facer_sycl[iD*N+itzm] = predict_iD_it - hds * dprim_iD_2_it; 

            facel_sycl[ivx*N+it] = predict_ivz_it + hds * dprim_ivz_2_it;
            facer_sycl[ivx*N+itzm] = predict_ivz_it - hds * dprim_ivz_2_it;

            facel_sycl[ivy*N+it] = predict_ivx_it + hds * dprim_ivx_2_it;
            facer_sycl[ivy*N+itzm] = predict_ivx_it - hds * dprim_ivx_2_it;

            facel_sycl[ivz*N+it] = predict_ivy_it + hds * dprim_ivy_2_it;
            facer_sycl[ivz*N+itzm] = predict_ivy_it - hds * dprim_ivy_2_it;
        }).wait();
    }
}
// Courant condition to calculate next timestep
double Courant(SyclQueuemanager& mgr){
    size_t N = mgr.N_;
    size_t ndim = mgr.ndim_;

    double cs = mgr.cs_;
    double gamma = mgr.gamma_;
    double Cdt = mgr.Cdt_;

    size_t iD = mgr.iD_;
    size_t iE = mgr.iE_;
    size_t ivx = mgr.ivx_;
    size_t ivy = mgr.ivy_;
    size_t ivz = mgr.ivz_;

    bool isothermal = mgr.isothermal_;

    sycl::queue& q    = mgr.get_queue();    
    double* vars_sycl = mgr.get_vars_sycl();
    double* ds_sycl   = mgr.get_ds_sycl();
    double* d_max     = mgr.get_d_max();
    double* min_dt   = mgr.get_min_dt();
    if (!isothermal){
        double gamma_factor = gamma - 1.0;

        auto red_x = sycl::reduction(&d_max[0], sycl::maximum<double>());
        auto red_y = sycl::reduction(&d_max[1], sycl::maximum<double>());
        auto red_z = sycl::reduction(&d_max[2], sycl::maximum<double>());

        q.parallel_for(sycl::range<1>(N), red_x, red_y, red_z, 
            [=](sycl::id<1> idx, auto& rx, auto& ry, auto& rz) {            
            size_t it = idx;
            d_max[0] = 0.0;
            d_max[1] = 0.0;
            d_max[2] = 0.0; 
            double D = vars_sycl[iD*N+it];
            double mom_x = vars_sycl[ivx*N+it];
            double mom_y = vars_sycl[ivy*N+it];
            double mom_z = vars_sycl[ivz*N+it];

            double E_kin = mom_x*mom_x + mom_y*mom_y + mom_z*mom_z;
            E_kin /= 2.0 * D;
            
            double pressure = gamma_factor * (vars_sycl[iE*N+it] - E_kin);
            // compute sound speed directly from pressure
            double cs = std::sqrt(gamma * pressure / D);

            double vx = std::abs(vars_sycl[ivx*N+it]) / D;
            rx.combine(vx + cs);
            double vy = std::abs(vars_sycl[ivy*N+it]) / D;
            ry.combine(vy + cs);
            double vz = std::abs(vars_sycl[ivz*N+it]) / D;
            rz.combine(vz + cs);
        }).wait();
    } else {
        auto red_x = sycl::reduction(&d_max[0], sycl::maximum<double>());
        auto red_y = sycl::reduction(&d_max[1], sycl::maximum<double>());
        auto red_z = sycl::reduction(&d_max[2], sycl::maximum<double>());
        double cs_ = cs;
        q.parallel_for(sycl::range<1>(N), red_x, red_y, red_z, 
            [=](sycl::id<1> idx, auto& rx, auto& ry, auto& rz) {            
            size_t it = idx;
            double D = vars_sycl[iD*N+it];
            d_max[0] = 0.0;
            d_max[1] = 0.0;
            d_max[2] = 0.0; 
            double vx = std::abs(vars_sycl[ivx*N+it]) / D;
            rx.combine(vx + cs_);
            double vy = std::abs(vars_sycl[ivy*N+it]) / D;
            ry.combine(vy + cs_);
            double vz = std::abs(vars_sycl[ivz*N+it]) / D;
            rz.combine(vz + cs_);
        }).wait();
    }    
    q.single_task([=]() {
        double local_min = 1e30;

        if (d_max[0] > 1e-15) {
            double dt_dim = ds_sycl[0] / d_max[0];
            if (dt_dim < local_min) local_min = dt_dim;
        }
        if (d_max[1] > 1e-15) {
            double dt_dim = ds_sycl[1] / d_max[1];
            if (dt_dim < local_min) local_min = dt_dim;
        }
        if (d_max[2] > 1e-15) {
            double dt_dim = ds_sycl[2] / d_max[2];
            if (dt_dim < local_min) local_min = dt_dim;
        }

        *min_dt = local_min;
    }).wait();

    double min_dt_c = 0.0;
    q.memcpy(&min_dt_c, min_dt, sizeof(double)).wait();
    
    double dt = Cdt * min_dt_c;
    return dt;
    }    

void Calc_Step(SyclQueuemanager& mgr, double dt){
    size_t dim0 = mgr.dim0_;
    size_t dim1 = mgr.dim1_;
    size_t dim2 = mgr.dim2_;
    size_t nv = mgr.nv_;
    
    size_t N = mgr.N_;
    size_t N_nv = nv*dim0*dim1*dim2;       // offset in 4D array for slope to go through correctly
    size_t ndim = mgr.ndim_;

    double cs = mgr.cs_;
    double gamma = mgr.gamma_;
    double Cdt = mgr.Cdt_;

    size_t iD = mgr.iD_;
    size_t iE = mgr.iE_;
    size_t ivx = mgr.ivx_;
    size_t ivy = mgr.ivy_;
    size_t ivz = mgr.ivz_;

    bool isothermal = mgr.isothermal_;
    
    // 1) Primitive variables D,P,v -- @t and cell centered
    //get pointers from SyclQueueManager
    sycl::queue& q = mgr.get_queue();
    double* vars_sycl    = mgr.get_vars_sycl();
    double* prim_sycl    = mgr.get_prim_sycl();
    double* dprim_sycl   = mgr.get_dprim_sycl();
    double* facel_sycl   = mgr.get_facel_sycl();
    double* facer_sycl   = mgr.get_facer_sycl();
    double* flux_sycl    = mgr.get_flux_sycl();
    double* invD_sycl    = mgr.get_invD_sycl();
    double* predict_sycl = mgr.get_predict_sycl();
    double* ds_sycl      = mgr.get_ds_sycl();
    double* ptr_ds       = mgr.get_ptr_ds();
    int* iw1             = mgr.get_iw_order_1();
    int* iw2             = mgr.get_iw_order_2();
    int* iF1             = mgr.get_iF_order_1();
    int* iF2             = mgr.get_iF_order_2();

    if (!isothermal) {
        double cs_factor = cs*cs;   
        double gamma_factor = gamma - 1.0;

        q.parallel_for(N, [=](sycl::id<1> it){
            double D = vars_sycl[iD*N+it];
            double invD = 1 / vars_sycl[iD*N+it];
            invD_sycl[it] = invD;
            double mom_x = vars_sycl[ivx*N+it];
            double mom_y = vars_sycl[ivy*N+it];
            double mom_z = vars_sycl[ivz*N+it];
            double E_kin = mom_x*mom_x + mom_y*mom_y + mom_z*mom_z;
            E_kin /= 2.0 * D;
            
            double P = gamma_factor * (vars_sycl[iE*N+it] - E_kin);
            
            prim_sycl[iD*N+it] = vars_sycl[iD*N+it];
            prim_sycl[iE*N+it] = P;
            prim_sycl[ivx*N+it] = vars_sycl[ivx*N+it] * invD;
            prim_sycl[ivy*N+it] = vars_sycl[ivy*N+it] * invD;
            prim_sycl[ivz*N+it] = vars_sycl[ivz*N+it] * invD;
        }).wait();
    } else {
        q.parallel_for(N, [=](sycl::id<1> it){
            double D = vars_sycl[iD*N+it];
            double invD = 1 / vars_sycl[iD*N+it];
            invD_sycl[it] = invD;
            double mom_x = vars_sycl[ivx*N+it];
            double mom_y = vars_sycl[ivy*N+it];
            double mom_z = vars_sycl[ivz*N+it];

            prim_sycl[iD*N+it] = vars_sycl[iD*N+it];
            prim_sycl[ivx*N+it] = vars_sycl[ivx*N+it] * invD;
            prim_sycl[ivy*N+it] = vars_sycl[ivy*N+it] * invD;
            prim_sycl[ivz*N+it] = vars_sycl[ivz*N+it] * invD;
        }).wait();
    }
    // 2) Slopes for primitive variables shape (nv,ndim,n,n,n) -- @t and cell centered
    moncen_sycl(q, prim_sycl, dprim_sycl, ds_sycl, nv, dim0, dim1, dim2);

    // 3) predicted solution @t+dt/2 shape (nv,n,n,n)

    // for each idim:   
    //   4) left and right face values (+-ds/2) @t+dt/2 _at_cell_interface_ with shape (2,nv,n,n,n)
    //   5) Reorder variables with the perpedicular velocity component as first index.
    //   6) Update the conserved variables

    // step 3 and 4 are fused in calc_faces_0/1/2
    size_t idim = 0;
    calc_faces_0(q, facel_sycl, facer_sycl, prim_sycl, dprim_sycl, invD_sycl, ds_sycl, N, iD, iE, ivx, ivy, ivz, ndim, dim0, dim1, dim2, isothermal, cs, gamma, dt);
    
    double* flux_sycl_ = HLL_combine(q, flux_sycl, facel_sycl, facer_sycl, N, nv, iD, iE, ivx, ivy, ivz, isothermal, cs, gamma);
    
    q.parallel_for(sycl::range<2>(nv, N), [=](sycl::id<2> idx){
        size_t iv = idx[0];
        size_t it = idx[1];
        double dt_ds = dt / ds_sycl[0];
        
        // convert to 3D spatial coordinates
        size_t iz = it % dim2;
        size_t iy = (it/ dim2) % dim1;
        size_t ix = it / (dim1 * dim2);

        size_t ixm = (ix + dim0 - 1) % dim0;
        size_t itxm = ixm * (dim1 * dim2) + iy * dim2 + iz;

        vars_sycl[iv*N+it] -= dt_ds*(flux_sycl_[iv*N+it] - flux_sycl_[iv*N+itxm]);
    }).wait();

    idim = 1;
    calc_faces_1(q, facel_sycl, facer_sycl, prim_sycl, dprim_sycl, invD_sycl, ds_sycl, N, iD, iE, ivx, ivy, ivz, ndim, dim0, dim1, dim2, isothermal, cs, gamma, dt);

    double* flux_sycl_1 = HLL_combine(q, flux_sycl, facel_sycl, facer_sycl, N, nv, iD, iE, ivx, ivy, ivz, isothermal, cs, gamma);

    q.parallel_for(sycl::range<2>(nv, N), [=](sycl::id<2> idx){    
        size_t i = idx[0];
        size_t iv = iF1[i];
        size_t it = idx[1];
        double dt_ds = dt / ds_sycl[1];

        // convert to 3D spatial coordinates
        size_t iz = it % dim2;
        size_t iy = (it/ dim2) % dim1;
        size_t ix = it / (dim1 * dim2);

        size_t iym = (iy + dim1 - 1) % dim1;
        size_t itym = ix * (dim1 * dim2) + iym * dim2 + iz;

        vars_sycl[i*N+it] -= dt_ds*(flux_sycl_1[iv*N+it] - flux_sycl_1[iv*N+itym]);
    }).wait();

    idim = 2;
    calc_faces_2(q, facel_sycl, facer_sycl, prim_sycl, dprim_sycl, invD_sycl, ds_sycl, N, iD, iE, ivx, ivy, ivz, ndim, dim0, dim1, dim2, isothermal, cs, gamma, dt);

    double* flux_sycl_2 = HLL_combine(q, flux_sycl, facel_sycl, facer_sycl, N, nv, iD, iE, ivx, ivy, ivz, isothermal, cs, gamma);

    q.parallel_for(sycl::range<2>(nv, N), [=](sycl::id<2> idx){    
        size_t i = idx[0];
        size_t iv = iF2[i];
        size_t it = idx[1];
        double dt_ds = dt / ds_sycl[2];

        // convert to 3D spatial coordinates
        size_t iz = it % dim2;
        size_t iy = (it/ dim2) % dim1;
        size_t ix = it / (dim1 * dim2);

        size_t izm = (iz + dim2 - 1) % dim2;
        size_t itzm = ix * (dim1 * dim2) + iy * dim2 + izm;

        vars_sycl[i*N+it] -= dt_ds*(flux_sycl_2[iv*N+it] - flux_sycl_2[iv*N+itzm]);
    }).wait();
}

PYBIND11_MODULE(muscl_step, m){
    m.doc() = "Full MUSCL solver for GPUs using SYCL";
    
    m.def("Calc_Step", &Calc_Step, "calculate all steps at once");
    m.def("Courant", &Courant, "calculate dt with Courant condition");
    py::class_<SyclQueuemanager>(m, "SyclQueuemanager")
    .def(py::init<>())  // bind the constructor
    .def("get_device_info", &SyclQueuemanager::get_device_info)
    .def("init_params", &SyclQueuemanager::init_params)
    .def("cpy_vars_from_device", &SyclQueuemanager::cpy_vars_from_device)
    .def("init_vars_from_host", &SyclQueuemanager::init_vars_from_host)
    .def("malloc_arrays", &SyclQueuemanager::malloc_arrays)
    .def("free_arrays", &SyclQueuemanager::free_arrays);
}