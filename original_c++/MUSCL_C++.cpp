#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// template function that works for differnet types
template <typename T> T* get_array_ptr(py::array_t< T > a){

    py::buffer_info a_buf = a.request();
    T* a_ptr = static_cast< T* >(a_buf.ptr);
    return a_ptr;
}
/*
with the original array, the array buffer, offset to first element of slice and number of skipped dimensions
makes a slice/view without copying over
use optional argument to make slice --> example: shape=(5,64,64,64) -> shape=(3,64,64,64)  

*/

py::array_t<double> make_slice(py::array_t<double> &arr, py::buffer_info &arr_buf, size_t ptr_offset, size_t dim_skipped, std::optional<ssize_t> override_first_dim = std::nullopt){

    double* ptr = static_cast<double*>(arr_buf.ptr) + ptr_offset;
    std::vector<ssize_t> shape(arr_buf.shape.begin() + dim_skipped, arr_buf.shape.end());
    std::vector<ssize_t> strides(arr_buf.strides.begin() + dim_skipped, arr_buf.strides.end());
    
    if (override_first_dim.has_value()) 
        shape[0] = override_first_dim.value();
    
    py::array view = py::array(
        py::dtype("float64"),
        shape,
        strides,
        ptr,
        arr
    );
    return view;
    }

py::array_t<double> pressure(py::array_t<double> D, py::array_t<double> E, py::array_t<double> M, py::tuple n, bool isothermal, double cs, double gamma){
    
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();

    std::vector<ssize_t> grid_shape = {dim0, dim1, dim2};
    py::array_t<double> result(grid_shape);
    auto result_acc = result.mutable_unchecked<3>();

    auto E_acc = E.unchecked<3>();
    auto D_acc = D.unchecked<3>();
    auto M_acc = M.unchecked<4>();

    double cs_factor = cs*cs;
    double gamma_factor = gamma - 1;
    if (isothermal){
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz)
        result_acc(ix,iy,iz) = cs_factor * D_acc(ix,iy,iz);
    } else {
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz){
        double E_kin = M_acc(0,ix,iy,iz)*M_acc(0,ix,iy,iz);
        E_kin += M_acc(1,ix,iy,iz)*M_acc(1,ix,iy,iz);
        E_kin += M_acc(2,ix,iy,iz)*M_acc(2,ix,iy,iz);
        double rho = D_acc(ix,iy,iz);
        
        E_kin /= 2*rho;
    
        result_acc(ix,iy,iz) = gamma_factor * (E_acc(ix,iy,iz) - E_kin);
        }
    }
    return result;
}
py::array_t<double> v_sound(py::array_t<double> D, py::array_t<double> E, py::array_t<double> M, py::tuple n, bool isothermal, double cs, double gamma){
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();

    py::array_t<double> p = pressure(D, E, M, n, isothermal, cs, gamma);
    std::vector<ssize_t> v_s_shape = {dim0, dim1, dim2};
    py::array_t<double> vs(v_s_shape);
    auto vs_acc = vs.mutable_unchecked<3>();

    auto p_acc = p.unchecked<3>();
    auto D_acc = D.unchecked<3>();
   
    if (!isothermal){
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz)
        vs_acc(ix,iy,iz) = std::sqrt(gamma * p_acc(ix,iy,iz)/D_acc(ix,iy,iz));
    } else {
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz)
        vs_acc(ix,iy,iz) = cs;
    }
    
    return vs;
}
py::array_t<double> MonCen(py::array_t<double> f_arr, py::array_t<double> slopes_arr) {
    py::buffer_info f_buf = f_arr.request();        // get buffers
    py::buffer_info slopes_buf = slopes_arr.request();

    auto f_dim = f_buf.ndim;        // get ndim and shape of f and slopes
    auto s = f_buf.shape;
    auto slopes_dim = slopes_buf.ndim;

    size_t dim0 = f_buf.shape[0];
    size_t dim1 = f_buf.shape[1];
    size_t dim2 = f_buf.shape[2];
    auto f = f_arr.unchecked<3>();          // make direct access arrays 
    auto slopes = slopes_arr.mutable_unchecked<4>();

    for (size_t ix=0; ix < dim0; ++ix) {        // loops through first dim
        size_t ixp = (ix+1) % dim0;              // makes ix+1 wraps around last index
        size_t ixm = (ix + dim0 - 1) % dim0;      // makes ix-1 wrap around correctly
        for (size_t iy=0; iy < dim1; ++iy) {
            size_t iyp = (iy+1) % dim1;
            size_t iym = (iy + dim1 - 1) % dim1;
            for (size_t iz=0; iz < dim2; ++iz) {
                size_t izp = (iz+1) % dim2;
                size_t izm = (iz + dim2 - 1) % dim2;

                double lsx = 0.5*(f(ix,iy,iz) - f(ixm,iy,iz));
                double rsx = 0.5*(f(ixp,iy,iz) - f(ix,iy,iz));
                double wx = lsx*rsx;   
                slopes(0,ix,iy,iz) = (wx > 0) ? (2.0*wx / (lsx+rsx)) : 0.0;      
                double lsy = 0.5*(f(ix,iy,iz) - f(ix,iym,iz));
                double rsy = 0.5*(f(ix,iyp,iz) - f(ix,iy,iz));
                double wy = lsy*rsy;
                slopes(1,ix,iy,iz) = (wy > 0) ? (2.0*wy / (lsy+rsy)) : 0.0;                    
                double lsz = 0.5*(f(ix,iy,iz) - f(ix,iy,izm));
                double rsz= 0.5*(f(ix,iy,izp) - f(ix,iy,iz));
                double wz = lsz*rsz;                    
                slopes(2,ix,iy,iz) = (wz > 0) ? (2.0*wz / (lsz+rsz)) : 0.0;
            }
        }
    }
    return slopes_arr; 
}
// Convert primitive variables to conserved variables
py::array_t<double> prim2cons(py::array_t<double> q, py::tuple n, int nv, int iD, int iE, bool isothermal, py::slice iM, double gamma){
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();

    std::vector<ssize_t> c_shape = {nv, dim0, dim1, dim2};      // initilize conservitive vars array
    py::array_t<double> c(c_shape);
    py::buffer_info q_buf = q.request();
    py::buffer_info c_buf = c.request();

    ssize_t start, stop, step, slice_length;    // initilize variables for the python slice
    ssize_t len_dim = q_buf.shape[0];
    if (!iM.compute(len_dim, &start, &stop, &step, &slice_length))  // make iM slice into usable variables
        throw std::runtime_error("Invalid slice");
    
    size_t ivx = start;
    size_t ivy = start + 1;
    size_t ivz = start + 2; 
    // make direct access arrays
    auto q_acc = q.unchecked<4>();
    auto c_acc = c.mutable_unchecked<4>();

    double factor = 1/(gamma-1);
    for (size_t ix=0; ix < dim0; ++ix)
    for (size_t iy=0; iy < dim1; ++iy)
        if (!isothermal){
        for (size_t iz=0; iz < dim2; ++iz){
            //  c[m.iD] = q[m.iD]       # Density
            c_acc(iD,ix,iy,iz) = q_acc(iD,ix,iy,iz);
            // compute total energy if not isothermal inline
            double Etot = q_acc(ivx,ix,iy,iz)*q_acc(ivx,ix,iy,iz) + q_acc(ivy,ix,iy,iz)*q_acc(ivy,ix,iy,iz) + q_acc(ivz,ix,iy,iz)*q_acc(ivz,ix,iy,iz);
            Etot *= 0.5*q_acc(iD,ix,iy,iz);
            Etot += factor * q_acc(iE,ix,iy,iz);
            c_acc(iE,ix,iy,iz) = Etot;
            c_acc(ivx,ix,iy,iz) = q_acc(iD,ix,iy,iz)*q_acc(ivx,ix,iy,iz);   // momentum
            c_acc(ivy,ix,iy,iz) = q_acc(iD,ix,iy,iz)*q_acc(ivy,ix,iy,iz);
            c_acc(ivz,ix,iy,iz) = q_acc(iD,ix,iy,iz)*q_acc(ivz,ix,iy,iz);
            }
        } else {
            for (size_t iz=0; iz < dim2; ++iz){
                c_acc(iD,ix,iy,iz) = q_acc(iD,ix,iy,iz);
                c_acc(ivx,ix,iy,iz) = q_acc(iD,ix,iy,iz)*q_acc(ivx,ix,iy,iz);
                c_acc(ivy,ix,iy,iz) = q_acc(iD,ix,iy,iz)*q_acc(ivy,ix,iy,iz);
                c_acc(ivz,ix,iy,iz) = q_acc(iD,ix,iy,iz)*q_acc(ivz,ix,iy,iz);
                }
            }
    return c;
}
// Compute hydro flux for conserved variables 
py::array_t<double> hydro_flux(py::array_t<double> c, py::array_t<double> q, py::tuple n, int nv, int iD, int iE, bool isothermal, py::slice iM, double cs){
    // convert from q to c with density, total energy and momentum
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();
    
    std::vector<ssize_t> F_shape = {nv, dim0, dim1, dim2};  // initilize conservitive vars array
    py::array_t<double> F(F_shape);

    py::buffer_info F_buf = F.request();
    py::buffer_info q_buf = q.request();
    py::buffer_info c_buf = c.request();

    ssize_t start, stop, step, slice_length;    // initilize variables for the python slice
    ssize_t len_dim = F_buf.shape[0];
    if (!iM.compute(len_dim, &start, &stop, &step, &slice_length))
        throw std::runtime_error("Invalid slice");
    size_t ivx = start;
    size_t ivy = start + 1;
    size_t ivz = start + 2;    

    auto F_acc = F.mutable_unchecked<4>();
    auto q_acc = q.unchecked<4>();
    auto c_acc = c.unchecked<4>();

    double factor = 2*cs*cs;
    for (size_t ix=0; ix < dim0; ++ix)
    for (size_t iy=0; iy < dim1; ++iy)
        if (!isothermal){
        for (size_t iz=0; iz < dim2; ++iz){
            F_acc(iD,ix,iy,iz) = c_acc(ivx,ix,iy,iz);     // Normal velocity v = q[iMl[0]]
            F_acc(ivx,ix,iy,iz) = c_acc(ivx,ix,iy,iz)*q_acc(ivx,ix,iy,iz);    // Velocity part of momentum flux F_v[i] = D v[i] v_norm
            F_acc(ivy,ix,iy,iz) = c_acc(ivy,ix,iy,iz)*q_acc(ivx,ix,iy,iz); 
            F_acc(ivz,ix,iy,iz) = c_acc(ivz,ix,iy,iz)*q_acc(ivx,ix,iy,iz); 
            F_acc(iE,ix,iy,iz) = (c_acc(iE,ix,iy,iz)+q_acc(iE,ix,iy,iz))*q_acc(ivx,ix,iy,iz);     // Energy flux = (E + P) v
            F_acc(ivx,ix,iy,iz) += q_acc(iE,ix,iy,iz);    // Add pressure to normal part of momentum flux
            }
        } else {
            for (size_t iz=0; iz < dim2; ++iz){
                F_acc(iD,ix,iy,iz) = c_acc(ivx,ix,iy,iz);
                F_acc(ivx,ix,iy,iz) = c_acc(ivx,ix,iy,iz)*q_acc(ivx,ix,iy,iz); 
                F_acc(ivy,ix,iy,iz) = c_acc(ivy,ix,iy,iz)*q_acc(ivx,ix,iy,iz); 
                F_acc(ivz,ix,iy,iz) = c_acc(ivz,ix,iy,iz)*q_acc(ivx,ix,iy,iz); 
                F_acc(ivx,ix,iy,iz) = factor*c_acc(iD,ix,iy,iz); 
                }
            }
    return F;
}
/*
Solve for the HLL flux given two HD state vectors (left and right of the interface)
        q = [left,right][Density, Pressure, vO, v1, v2], where vO is normal to the interface
        between left and right state, and v1, v2 are parallel to the interface
*/
py::array_t<double> HLL(py::array_t<double> ql, py::array_t<double> qr, py::tuple n, int nv, int iD, int iE, py::slice iM, bool isothermal, double cs, double gamma){
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();

    ssize_t start, stop, step, slice_length;
    ssize_t len_dim = ql.request().shape[0];
    if (!iM.compute(len_dim, &start, &stop, &step, &slice_length))
        throw std::runtime_error("Invalid slice");
    size_t ivx = start;

    py::array_t<double> cl = prim2cons(ql, n, nv, iD, iE, isothermal, iM, gamma);   // Convert to conserved variables       
    py::array_t<double> cr = prim2cons(qr, n, nv, iD, iE, isothermal, iM, gamma);
    py::array_t<double> Fl = hydro_flux(cl, ql, n, nv, iD, iE, isothermal, iM, cs);   // Compute hydro flux
    py::array_t<double> Fr = hydro_flux(cr, qr, n, nv, iD, iE, isothermal, iM, cs);
    
    std::vector<ssize_t> flux_shape = {nv, dim0, dim1, dim2};   // initilize flux
    py::array_t<double> flux(flux_shape);

    auto flux_acc = flux.mutable_unchecked<4>();
    auto ql_acc = ql.unchecked<4>();
    auto qr_acc = qr.unchecked<4>();
    auto cl_acc = cl.unchecked<4>();
    auto cr_acc = cr.unchecked<4>();
    auto Fl_acc = Fl.unchecked<4>();
    auto Fr_acc = Fr.unchecked<4>();
    
    double c2l;     // maximum signal speed
    double c2r;
    double c_max;

    std::vector<ssize_t> n_shape = {dim0, dim1, dim2};
    py::array_t<double> SL(n_shape);
    py::array_t<double> SR(n_shape);
    py::array_t<double> iSRL(n_shape);
    py::array_t<double> SRL(n_shape);
    
    auto SL_acc = SL.mutable_unchecked<3>();
    auto SR_acc = SR.mutable_unchecked<3>();
    auto iSRL_acc = iSRL.mutable_unchecked<3>();
    auto SRL_acc = SRL.mutable_unchecked<3>();

    if (!isothermal){
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz){
            c2l = gamma * ql_acc(iE,ix,iy,iz)/ql_acc(iD,ix,iy,iz);  
            c2r = gamma * qr_acc(iE,ix,iy,iz)/qr_acc(iD,ix,iy,iz);  
            c_max = std::sqrt(std::max(c2l, c2r));
            
            SL_acc(ix,iy,iz) = std::min(std::min(ql_acc(ivx,ix,iy,iz), qr_acc(ivx,ix,iy,iz)) - c_max, 0.0);
            SR_acc(ix,iy,iz) = std::max(std::max(ql_acc(ivx,ix,iy,iz), qr_acc(ivx,ix,iy,iz)) + c_max, 0.0);
            iSRL_acc(ix,iy,iz) = 1/(SR_acc(ix,iy,iz) - SL_acc(ix,iy,iz));
            SRL_acc(ix,iy,iz) = SR_acc(ix,iy,iz)*SL_acc(ix,iy,iz);
        }
        for (size_t iv=0; iv < nv; ++iv)
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz){
            double SL = SL_acc(ix,iy,iz);
            double SR = SR_acc(ix,iy,iz);
            double iSRL = iSRL_acc(ix,iy,iz);
            double SRL = SRL_acc(ix,iy,iz);
            
            flux_acc(iv,ix,iy,iz) = (SR*Fl_acc(iv,ix,iy,iz) - SL*Fr_acc(iv,ix,iy,iz) + SRL*(cr_acc(iv,ix,iy,iz) - cl_acc(iv,ix,iy,iz)))*iSRL;
        }

        } else {
            for (size_t ix=0; ix < dim0; ++ix)
            for (size_t iy=0; iy < dim1; ++iy)
            for (size_t iz=0; iz < dim2; ++iz){
                c_max = cs;
                
                SL_acc(ix,iy,iz) = std::min(std::min(ql_acc(ivx,ix,iy,iz), qr_acc(ivx,ix,iy,iz)) - c_max, 0.0);
                SR_acc(ix,iy,iz) = std::max(std::max(ql_acc(ivx,ix,iy,iz), qr_acc(ivx,ix,iy,iz)) + c_max, 0.0);
                iSRL_acc(ix,iy,iz) = 1/(SR_acc(ix,iy,iz) - SL_acc(ix,iy,iz));
                SRL_acc(ix,iy,iz) = SR_acc(ix,iy,iz)*SL_acc(ix,iy,iz);
            }
            for (size_t iv=0; iv < nv; ++iv)
            for (size_t ix=0; ix < dim0; ++ix)
            for (size_t iy=0; iy < dim1; ++iy)
            for (size_t iz=0; iz < dim2; ++iz){
                double SL = SL_acc(ix,iy,iz);
                double SR = SR_acc(ix,iy,iz);
                double iSRL = iSRL_acc(ix,iy,iz);
                double SRL = SRL_acc(ix,iy,iz);
                
                flux_acc(iv,ix,iy,iz) = (SR*Fl_acc(iv,ix,iy,iz) - SL*Fr_acc(iv,ix,iy,iz) + SRL*(cr_acc(iv,ix,iy,iz) - cl_acc(iv,ix,iy,iz)))*iSRL;
            }
        }
    return flux;
}
double Courant(py::array_t<double> D, py::array_t<double> E, py::array_t<double> M, py::array_t<double> ds, py::tuple n, bool isothermal, ssize_t ndim, double cs_, double gamma, double Cdt){
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();

    std::vector<ssize_t> shape = {dim0, dim1, dim2};
    py::array_t<double> cs(shape); 
    double* ptr_cs = get_array_ptr(cs);

    if (isothermal){
        double total_ele = dim0*dim1*dim2;
        for (size_t n=0; n < total_ele; ++n)
            ptr_cs[n] = cs_;
    } else cs = v_sound(D, E, M, n, isothermal, cs_, gamma).cast<py::array_t<double>>();
    
    auto D_acc = D.unchecked<3>();
    auto M_acc = M.unchecked<4>();
    auto ds_acc = ds.unchecked<1>();
    auto cs_acc = cs.mutable_unchecked<3>();

    std::vector<double> vmax(ndim, 0.0);
    for (size_t idim=0; idim<ndim; ++idim){
        double max_val = 0.0;
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz){
            double result = std::abs(M_acc(idim,ix,iy,iz)) / D_acc(ix,iy,iz);
            double v = cs_acc(ix,iy,iz) + result;
            if (v > max_val){
                max_val = v;  
                }
            }  
        vmax[idim] = max_val; 
    }
    double min_dt = 1e30; 
    for (size_t idim=0; idim<ndim; ++idim){
        double dt_dim = ds_acc(idim) / vmax[idim];
        if (dt_dim < min_dt) min_dt = dt_dim; 
    }
    double dt = Cdt * min_dt;
    return dt;
    }

/*
    muscl step function:
    0. Compute the Courant condition
    1. Primitive variables D,P,v -- @t and cell centered
    2. Slopes for primitive variables shape (nv,ndim,n,n,n) -- @t and cell centered
    3. predicted solution @t+dt/2 shape (nv,n,n,n)
    4. left and right face values (+-ds/2) @t+dt/2 _at_cell_interface_ with shape (2,nv,n,n,n)
    5. Reorder variables with the perpedicular velocity component as first index.
    6. Update the conserved variables
*/
void Calc_Step(py::array_t<double> vars, py::array_t<double> D, py::array_t<double> E, py::array_t<double> M, py::array_t<double> ds, py::tuple n, ssize_t nv, ssize_t ndim, ssize_t iD, ssize_t iE, py::slice iM, double dt, double Cdt, double cs, double gamma, bool isothermal){
    ssize_t dim0 = n[0].cast<size_t>();    // get dimensions from grid resolution
    ssize_t dim1 = n[1].cast<size_t>();
    ssize_t dim2 = n[2].cast<size_t>();

    ssize_t start, stop, step, slice_length;    // initilize variables for the python slice
    if (!iM.compute(nv, &start, &stop, &step, &slice_length))   // make iM slice into usable variables
        throw std::runtime_error("Invalid slice");
    
    size_t ivx = start;         // indicies for Momentum 
    size_t ivy = start + 1;
    size_t ivz = start + 2;    

    size_t offset_3D = dim0*dim1*dim2;
    size_t offset_4D = ndim*dim0*dim1*dim2;       // offset in 4D array for slope to go through correctly
    size_t skipped_dim = 1;         

    // shapes of coming arrays
    std::vector<ssize_t> grid_shape = {dim0, dim1, dim2};
    std::vector<ssize_t> nv_shape = {nv, dim0, dim1, dim2};
    std::vector<ssize_t> v_shape = {nv, ndim, dim0, dim1, dim2};

    // create all neccessary arrays
    py::array_t<double> prim(nv_shape);
    py::array_t<double> dprim(v_shape);
    py::array_t<double> predict(nv_shape);
    py::array_t<double> facel(nv_shape);
    py::array_t<double> facer(nv_shape);
    py::array_t<double> flux_(nv_shape);

    py::array_t<double> P = pressure(D, E, M, n, isothermal, cs, gamma);    // calc pressure

    // make all buffers
    py::buffer_info prim_buf = prim.request();    
    py::buffer_info dprim_buf = dprim.request();
    
    // make all neccessary pointers
    double* ptr_prim =  get_array_ptr(prim);     
    double* ptr_dprim = get_array_ptr(dprim);
    double* ptr_ds =    get_array_ptr(ds);

    // make all direct access arrays
    auto vars_acc = vars.mutable_unchecked<4>();
    auto prim_acc = prim.mutable_unchecked<4>();
    auto dprim_acc = dprim.unchecked<5>();
    auto predict_acc = predict.mutable_unchecked<4>();
    auto facel_acc = facel.mutable_unchecked<4>();
    auto facer_acc = facer.mutable_unchecked<4>();    
    auto M_acc = M.mutable_unchecked<4>();
    auto D_acc = D.unchecked<3>();
    auto P_acc = P.unchecked<3>();
    auto ds_acc = ds.unchecked<1>();

    // 1) Primitive variables D,P,v -- @t and cell centered
    if (!isothermal){
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz){
            double invD = 1 / D_acc(ix,iy,iz);
            prim_acc(iD,ix,iy,iz) = D_acc(ix,iy,iz);                // add Density
            prim_acc(iE,ix,iy,iz) = P_acc(ix,iy,iz);                // add Energy
            prim_acc(ivx,ix,iy,iz) = M_acc(0,ix,iy,iz) * invD;    // calc velocity
            prim_acc(ivy,ix,iy,iz) = M_acc(1,ix,iy,iz) * invD;
            prim_acc(ivz,ix,iy,iz) = M_acc(2,ix,iy,iz) * invD;
        }
    } else {
        for (size_t ix=0; ix < dim0; ++ix)
        for (size_t iy=0; iy < dim1; ++iy)
        for (size_t iz=0; iz < dim2; ++iz){
            double invD = 1 / D_acc(ix,iy,iz);
            prim_acc(iD,ix,iy,iz) = D_acc(ix,iy,iz);                // add Density
            prim_acc(ivx,ix,iy,iz) = M_acc(0,ix,iy,iz) * invD;    // calc velocity
            prim_acc(ivy,ix,iy,iz) = M_acc(1,ix,iy,iz) * invD;
            prim_acc(ivz,ix,iy,iz) = M_acc(2,ix,iy,iz) * invD;
        }
    }

    // 2) Slopes for primitive variables shape (nv,ndim,n,n,n) -- @t and cell centered
    size_t total_elements = offset_3D;
    for (size_t iv=0; iv < nv; ++iv) {
        size_t prim_slice_offset = iv * prim_buf.strides[0] / sizeof(double);
        size_t dprim_slice_offset = iv * dprim_buf.strides[0] / sizeof(double);

        py::array_t<double> prim_slice = make_slice(prim, prim_buf, prim_slice_offset, skipped_dim);
        py::array_t<double> dprim_slice = make_slice(dprim, dprim_buf, dprim_slice_offset, skipped_dim);

        MonCen(prim_slice, dprim_slice);  // Pass new views od prim & dprim

        for (size_t idim=0; idim < ndim; ++idim){
            double inv_ds = 1.0 / ds_acc(idim);
            for (size_t n=0; n < total_elements; ++n)
            ptr_dprim[iv*offset_4D+idim*total_elements + n] *= inv_ds;
            }
        }
    // 3) predicted solution @t+dt/2 shape (nv,n,n,n)
    size_t D_offset = iD * prim_buf.strides[0] / sizeof(double);
    size_t dD_offset = iD * dprim_buf.strides[0] / sizeof(double);
    
    // make slices of dD and dP
    py::array_t<double> dD = make_slice(dprim, dprim_buf, dD_offset, skipped_dim);
    py::array_t<double> dP;
    if (isothermal){
        dP = py::array_t<double>(nv_shape);
        double* ptr_dD = get_array_ptr(dD);     
        double* ptr_dP = get_array_ptr(dP);

        double factor = cs*cs;  
        for (size_t n=0; n < offset_4D; ++n)    
            ptr_dP[n] = factor * ptr_dD[n];     // dP = m.cs**2 * dD
    } else {
        size_t dP_offset = iE * dprim_buf.strides[0] / sizeof(double);
        dP = make_slice(dprim, dprim_buf, dP_offset, skipped_dim);
    }
    // make offsets for v and dv
    size_t v_offset = start * prim_buf.strides[0] / sizeof(double);   
    size_t dv_0_offset = ivx * dprim_buf.strides[0] / sizeof(double);
    size_t dv_1_offset = ivy * dprim_buf.strides[0] / sizeof(double);
    size_t dv_2_offset = ivz * dprim_buf.strides[0] / sizeof(double);

    // make slice for v and dv
    py::array_t<double> v = make_slice(prim, prim_buf, v_offset, 0, slice_length);
    py::array_t<double> dv_0 = make_slice(dprim, dprim_buf, dv_0_offset, skipped_dim);
    py::array_t<double> dv_1 = make_slice(dprim, dprim_buf, dv_1_offset, skipped_dim);
    py::array_t<double> dv_2 = make_slice(dprim, dprim_buf, dv_2_offset, skipped_dim);
    
    
    // direct access arrays for step 3
    auto v_acc = v.unchecked<4>();
    auto dD_acc = dD.unchecked<4>();
    auto dP_acc = dP.unchecked<4>();
    auto dv_0_acc = dv_0.mutable_unchecked<4>();
    auto dv_1_acc = dv_1.mutable_unchecked<4>();
    auto dv_2_acc = dv_2.mutable_unchecked<4>();

    for (size_t ix=0; ix < dim0; ++ix)
    for (size_t iy=0; iy < dim1; ++iy)
        if (!isothermal){
            for (size_t iz=0; iz < dim2; ++iz){
                double invD = 1 / D_acc(ix,iy,iz);
                double div_v_trace = dv_0_acc(0,ix,iy,iz) + dv_1_acc(1,ix,iy,iz) + dv_2_acc(2,ix,iy,iz);

                double sum_v_dD = v_acc(0,ix,iy,iz)*dD_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dD_acc(1,ix,iy,iz) +v_acc(2,ix,iy,iz)*dD_acc(2,ix,iy,iz);
                predict_acc(iD,ix,iy,iz) = D_acc(ix,iy,iz) - 0.5*dt* (sum_v_dD + D_acc(ix,iy,iz)*div_v_trace);
                
                double sum_v_dP = v_acc(0,ix,iy,iz)*dP_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dP_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dP_acc(2,ix,iy,iz);
                predict_acc(iE,ix,iy,iz) = P_acc(ix,iy,iz) - 0.5*dt* (sum_v_dP + gamma*P_acc(ix,iy,iz)*div_v_trace);
                
                double sum_v_dv_0 = v_acc(0,ix,iy,iz)*dv_0_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dv_0_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dv_0_acc(2,ix,iy,iz);
                predict_acc(ivx,ix,iy,iz) = v_acc(0,ix,iy,iz) - 0.5*dt* (sum_v_dv_0 + invD*dP_acc(0,ix,iy,iz));

                double sum_v_dv_1 = v_acc(0,ix,iy,iz)*dv_1_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dv_1_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dv_1_acc(2,ix,iy,iz);
                predict_acc(ivy,ix,iy,iz) = v_acc(1,ix,iy,iz) - 0.5*dt* (sum_v_dv_1 + invD*dP_acc(1,ix,iy,iz));
                
                double sum_v_dv_2 = v_acc(0,ix,iy,iz)*dv_2_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dv_2_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dv_2_acc(2,ix,iy,iz);
                predict_acc(ivz,ix,iy,iz) = v_acc(2,ix,iy,iz) - 0.5*dt* (sum_v_dv_2 + invD*dP_acc(2,ix,iy,iz));
            }
        } else {
            for (size_t iz=0; iz < dim2; ++iz){
                double invD = 1 / D_acc(ix,iy,iz);
                double div_v_trace = dv_0_acc(0,ix,iy,iz) + dv_1_acc(1,ix,iy,iz) + dv_2_acc(2,ix,iy,iz);
    
                double sum_v_dD = v_acc(0,ix,iy,iz)*dD_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dD_acc(1,ix,iy,iz) +v_acc(2,ix,iy,iz)*dD_acc(2,ix,iy,iz);
                predict_acc(iD,ix,iy,iz) = D_acc(ix,iy,iz) - 0.5*dt* (sum_v_dD + D_acc(ix,iy,iz)*div_v_trace);
                
                double sum_v_dv_0 = v_acc(0,ix,iy,iz)*dv_0_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dv_0_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dv_0_acc(2,ix,iy,iz);
                predict_acc(ivx,ix,iy,iz) = v_acc(0,ix,iy,iz) - 0.5*dt* (sum_v_dv_0 + invD*dP_acc(0,ix,iy,iz));
    
                double sum_v_dv_1 = v_acc(0,ix,iy,iz)*dv_1_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dv_1_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dv_1_acc(2,ix,iy,iz);
                predict_acc(ivy,ix,iy,iz) = v_acc(1,ix,iy,iz) - 0.5*dt* (sum_v_dv_1 + invD*dP_acc(1,ix,iy,iz));
                
                double sum_v_dv_2 = v_acc(0,ix,iy,iz)*dv_2_acc(0,ix,iy,iz) + v_acc(1,ix,iy,iz)*dv_2_acc(1,ix,iy,iz) + v_acc(2,ix,iy,iz)*dv_2_acc(2,ix,iy,iz);
                predict_acc(ivz,ix,iy,iz) = v_acc(2,ix,iy,iz) - 0.5*dt* (sum_v_dv_2 + invD*dP_acc(2,ix,iy,iz));
                }
        }
        // for each idim:   
        //   4) left and right face values (+-ds/2) @t+dt/2 _at_cell_interface_ with shape (2,nv,n,n,n)
        //   5) Reorder variables with the perpedicular velocity component as first index.
        //   6) Update the conserved variables

        size_t idim = 0;
        double hds = 0.5*ds_acc(idim);
        for (size_t iv=0; iv<nv; ++iv)
        for (size_t ix=0; ix < dim0; ++ix){
            size_t ixp = (ix + 1) % dim0; 
            for (size_t iy=0; iy < dim1; ++iy)
            for (size_t iz=0; iz < dim2; ++iz){
                facel_acc(iv,ix,iy,iz) = predict_acc(iv,ix,iy,iz) + hds*dprim_acc(iv,idim,ix,iy,iz);
                facer_acc(iv,ix,iy,iz) = predict_acc(iv,ixp,iy,iz) - hds*dprim_acc(iv,idim,ixp,iy,iz);          
            }
        }
        flux_ = HLL(facel, facer, n, nv, iD, iE, iM, isothermal, cs, gamma);
        auto flux_acc = flux_.unchecked<4>();
        
        for (size_t iv=0; iv < nv; ++iv)
        for (size_t ix=0; ix < dim0; ++ix){
            size_t ixm = (ix + dim0 - 1) % dim0;      // makes ix-1 wrap around correctly
            for (size_t iy=0; iy < dim1; ++iy)
            for (size_t iz=0; iz < dim2; ++iz){
                vars_acc(iv,ix,iy,iz) -= (dt/ds_acc(idim))*(flux_acc(iv,ix,iy,iz) - flux_acc(iv,ixm,iy,iz));
            }
        }
        // indicies for correctly reordering for HLL solver
        // iw = 0,1,3,4,2
        // iF = 0,1,4,2,3
        idim = 1;
        std::vector<size_t> iw_order = {0, 1, 3, 4, 2};  // order for momentum vars 
        hds = 0.5*ds_acc(idim);
        for (size_t i=0; i<nv; ++i){
            size_t iv = iw_order[i];
            for (size_t ix=0; ix < dim0; ++ix)
            for (size_t iy=0; iy < dim1; ++iy){
                size_t iyp = (iy + 1) % dim1; 
                for (size_t iz=0; iz < dim2; ++iz){
                    facel_acc(i,ix,iy,iz) = predict_acc(iv,ix,iy,iz) + hds*dprim_acc(iv,idim,ix,iy,iz);
                    facer_acc(i,ix,iy,iz) = predict_acc(iv,ix,iyp,iz) - hds*dprim_acc(iv,idim,ix,iyp,iz);  
                }
            }
        }
        flux_ = HLL(facel, facer, n, nv, iD, iE, iM, isothermal, cs, gamma);
        auto flux_acc_1 = flux_.unchecked<4>();

        std::vector<size_t> iF_order = {0, 1, 4, 2, 3};  // reorder for momentum vars 
        for (size_t i=0; i < nv; ++i){
            size_t iv = iF_order[i];
            for (size_t ix=0; ix < dim0; ++ix)
            for (size_t iy=0; iy < dim1; ++iy){
                size_t iym = (iy + dim1 - 1) % dim1;      
                for (size_t iz=0; iz < dim2; ++iz){
                    vars_acc(i,ix,iy,iz) -= (dt/ds_acc(idim))*(flux_acc_1(iv,ix,iy,iz) - flux_acc_1(iv,ix,iym,iz));
                }
            }
        }
        // iw = 0,1,4,2,3
        // iF = 0,1,3,4,2
        idim = 2;
        hds = 0.5*ds_acc(idim);
        iw_order = {0, 1, 4, 2, 3};  // order for momentum vars 
        for (size_t i=0; i<nv; ++i){
            size_t iv = iw_order[i];
            for (size_t ix=0; ix < dim0; ++ix)
            for (size_t iy=0; iy < dim1; ++iy)
            for (size_t iz=0; iz < dim2; ++iz){
                size_t izp = (iz + 1) % dim2; 
                facel_acc(i,ix,iy,iz) = predict_acc(iv,ix,iy,iz) + hds*dprim_acc(iv,idim,ix,iy,iz);
                facer_acc(i,ix,iy,iz) = predict_acc(iv,ix,iy,izp) - hds*dprim_acc(iv,idim,ix,iy,izp);
            }
        }
        flux_ = HLL(facel, facer, n, nv, iD, iE, iM, isothermal, cs, gamma);
        auto flux_acc_2 = flux_.unchecked<4>();

        iF_order = {0, 1, 3, 4, 2};  // reorder for momentum vars 
        for (size_t i=0; i < nv; ++i){
            size_t iv = iF_order[i];
            for (size_t ix=0; ix < dim0; ++ix)
            for (size_t iy=0; iy < dim1; ++iy)
            for (size_t iz=0; iz < dim2; ++iz){
                size_t izm = (iz + dim2 - 1) % dim2;      
                vars_acc(i,ix,iy,iz) -= (dt/ds_acc(idim))*(flux_acc_2(iv,ix,iy,iz) - flux_acc_2(iv,ix,iy,izm));
            }
        }
std::cout<<"worked"<<std::endl;
}


PYBIND11_MODULE(muscl_step, m){
    m.doc() = "s";
    
    m.def("Calc_Step", &Calc_Step, "calculate all steps at once");
    m.def("Courant", &Courant, "calculate dt with Courant condition");
    }
