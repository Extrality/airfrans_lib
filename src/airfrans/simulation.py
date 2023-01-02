import os
import os.path as osp
import numpy as np
import pyvista as pv

from airfrans.reorganize import reorganize
from airfrans.naca_generator import camber_line
import airfrans.sampling as sampling

class Simulation:
    """
    Wrapper to make the study of AirfRANS simulations easier. It only takes in input the location of the
    ``.vtu`` and ``.vtp`` files from the internal and airfoil patches of the simulation.
    """
    def __init__(self, root, name, T = 298.15):
        """
        Define the air properties at temperature ``T``, the parameters, 
        input features and targets for the simulation.

        Args:
            root (str): Root directory where the dataset is.
            name (str): Name of the simulation.
            T (float, optional): Temperature set for the simulation in Kelvin (in incompressible settings, this only defines 
                the air properties. Simulations do not depend on temperature, only on the Reynolds number). Default: 298.15
        """

        self.root = root
        self.name = name
        self.T = np.array(T) # Temperature in Kelvin

        self.reset()
    
    def reset(self):        
        self.MOL = np.array(28.965338e-3) # Air molar weigth in kg/mol
        self.P_ref = np.array(1.01325e5) # Pressure reference in Pa
        self.RHO = self.P_ref*self.MOL/(8.3144621*self.T) # Specific mass of air at temperature T
        self.NU = -3.400747e-6 + 3.452139e-8*self.T + 1.00881778e-10*self.T**2 - 1.363528e-14*self.T**3 # Approximation of the kinematic viscosity of air at temperature T
        self.C = 20.05*np.sqrt(self.T) # Approximation of the sound velocity of air at temperature T        

        self.inlet_velocity = np.array(float(self.name.split('_')[2])).astype('float64')
        self.angle_of_attack = np.array(float(self.name.split('_')[3])*np.pi/180).astype('float64')

        self.internal = pv.read(osp.join(self.root, self.name, self.name + '_internal.vtu'))
        self.airfoil = pv.read(osp.join(self.root, self.name, self.name + '_aerofoil.vtp'))
        self.internal = self.internal.compute_cell_sizes(length = False, volume = False)
        self.airfoil = self.airfoil.compute_cell_sizes(area = False, volume = False)

        # Input candidates
        self.surface = np.array((self.internal.point_data['U'][:, 0] == 0))
        self.sdf = -np.array(self.internal.point_data['implicit_distance'][:, None]).astype('float64')
        self.input_velocity = (np.array([np.cos(self.angle_of_attack), np.sin(self.angle_of_attack)])*self.inlet_velocity).reshape(1, 2)*np.ones_like(self.sdf)
        
        self.position = np.array(self.internal.points[:, :2]).astype('float64')
        self.airfoil_position = np.array(self.airfoil.points[:, :2]).astype('float64')
        self.airfoil_normals = np.array(-self.airfoil.point_data['Normals'][:, :2]).astype('float64')

        self.normals = np.zeros_like(self.input_velocity).astype('float64')
        self.normals[self.surface] = reorganize(
            self.airfoil_position,
            self.position[self.surface], 
            self.airfoil_normals
            )

        # Targets
        self.velocity = np.array(self.internal.point_data['U'][:, :2]).astype('float64')
        self.pressure = np.array(self.internal.point_data['p'][:, None]).astype('float64')
        self.nu_t = np.array(self.internal.point_data['nut'][:, None]).astype('float64')

    def sampling_volume(self, n, density = 'uniform', targets = True):
        """
        Sample points in the internal mesh following the given density.

        Args:
            n (int): Number of sampled points.
            density (str, optional): Density from which the sampling is done. Choose between ``'uniform'`` and ``'mesh_density'``. Default: ``'uniform'``
            targets (bool, optional): If ``True``, velocity, pressure and kinematic turbulent viscosity will be returned in the output ndarray. Default: ``True``
        """
        if density == 'uniform': # Uniform sampling strategy
            p = np.array(self.internal.cell_data['Area']/self.internal.cell_data['Area'].sum())
            sampled_cell_indices = np.random.choice(self.internal.n_cells, size = n, replace = True, p = p)
        elif density == 'mesh_density': # Sample via mesh density
            sampled_cell_indices = np.random.choice(self.internal.n_cells, size = n, replace = True)

        cell_dict = np.array(self.internal.cells).reshape(-1, 5)[sampled_cell_indices, 1:]
        cell_points = self.position[cell_dict]

        if targets:
            cell_attr = np.concatenate([self.sdf[cell_dict], self.velocity[cell_dict], self.pressure[cell_dict], self.nu_t[cell_dict]], axis = -1)
        else:
            cell_attr = self.sdf[cell_dict]            

        return sampling.cell_sampling_2d(cell_points, cell_attr)
    
    def sampling_surface(self, n, density = 'uniform', targets = True):
        """
        Sample points in the airfoil mesh following the given density.

        Args:
            n (int): Number of sampled points.
            density (str, optional): Density from which the sampling is done. Choose between ``'uniform'`` and ``'mesh_density'``. Default: ``'uniform'``
            targets (bool, optional): If ``True``, velocity, pressure and kinematic turbulent viscosity will be returned in the output ndarray. Default: ``True``
        """
        if density == 'uniform': # Uniform sampling strategy
            p = np.array(self.airfoil.cell_data['Length']/self.airfoil.cell_data['Length'].sum())
            sampled_line_indices = np.random.choice(self.airfoil.n_cells, size = n, replace = True, p = p)
        elif density == 'mesh_density': # Sample via mesh density
            sampled_line_indices = np.random.choice(self.airfoil.n_cells, size = n, replace = True)

        line_dict = np.array(self.airfoil.lines).reshape(-1, 3)[sampled_line_indices, 1:]
        line_points = np.array(self.airfoil.points)[line_dict]

        normal = np.array(-self.airfoil.point_data['Normals'][line_dict, :2]) 

        if targets:
            line_attr = np.concatenate([
                normal, 
                np.array(self.airfoil.point_data['U'][line_dict, :2]), 
                np.array(self.airfoil.point_data['p'][line_dict, None]), 
                np.array(self.airfoil.point_data['nut'][line_dict, None])
                ], axis = -1)
        else:
            line_attr = normal     

        return sampling.cell_sampling_1d(line_points, line_attr)

    def sampling_mesh(self, n, targets = True):
        """
        Sample points over the simulation mesh without replacement.

        Args:
            n (int): Number of sampled points, this number has to be lower than the total number of points in the mesh.
            targets (bool, optional): If ``True``, velocity, pressure and kinematic turbulent viscosity will be returned in the output ndarray. Default: ``True``
        """
        idx = np.random.choice(self.position.shape[0], size = n, replace = False)

        # Position
        position = self.position[idx]

        # Inputs
        surface = self.surface[idx]
        sdf = self.sdf[idx]
        normals = self.normals[idx]
        input_velocity = self.input_velocity[idx]

        if targets:
            velocity = self.velocity[idx]
            pressure = self.pressure[idx]
            nu_t = self.nu_t[idx]
            sample = np.concatenate([position, surface[:, None], sdf, normals, input_velocity, velocity, pressure, nu_t], axis = -1)
        else:
            sample = np.concatenate([position, surface[:, None], sdf, normals, input_velocity], axis = -1)

        return sample

    def wallshearstress(self, over_airfoil = False, reference = False):
        """
        Compute the wall shear stress.

        Args:
            over_airfoil (bool, optional): If ``True``, return the wall shear stress over the airfoil mesh. If ``False``,
                return the wall shear stress over the internal mesh. Default: ``False``
            reference (bool, optional): If ``True``, return the wall shear stress computed with the reference velocity field.
                If ``False``, compute the wall shear stress with the velocity attribute of the class. Default: ``False``     
        """
        if reference:
            internal = self.internal.copy()
            internal.point_data['U'] = internal.point_data['U'].astype('float64')
        else:
            internal = self.internal.copy()
            internal.point_data['U'] = np.concatenate([self.velocity, np.zeros_like(self.sdf)], axis = -1)
        jacobian = np.array(internal.compute_derivative(scalars = 'U', gradient = 'jacobian').point_data['jacobian'].reshape(-1, 3, 3)[self.surface, :2, :2]).astype('float64')

        s = .5*(jacobian + jacobian.transpose(0, 2, 1))
        s = s - s.trace(axis1 = -2, axis2 = -1).reshape(-1, 1, 1)*np.eye(2)[None]/3

        wss_ = 2*self.NU.reshape(-1, 1, 1)*s
        wss_ = (wss_*self.normals[self.surface, :2].reshape(-1, 1, 2)).sum(axis = 2)

        if over_airfoil:
            wss = reorganize(self.position[self.surface], self.airfoil_position, wss_)
        else:
            wss = np.zeros_like(self.velocity).astype('float64')
            wss[self.surface] = wss_

        return wss
    
    def force(self, compressible = False, reference = False):
        """
        Compute the force acting on the airfoil. The output is a tuple of the form `(f, fp, fv)`, 
        where `f` is the force, `fp` the pressure contribution of the force and `fv` the viscous
        contribution of the force.

        Args:
            compressible (bool, optional): If ``False``, multiply the force computed with the simulation field by the specific mass. Default: ``False``
            reference (bool, optional): If ``True``, return the force computed with the reference fields.
                If ``False``, compute the force with the fields attribute of the class. Default: ``False``   
        """
        wss = self.wallshearstress(over_airfoil = True, reference = reference)

        if reference:
            p = np.array(self.internal.point_data['p'][:, None]).astype('float64')
        else:
            p = self.pressure
        p = reorganize(self.position[self.surface], self.airfoil_position, p[self.surface])

        airfoil = self.airfoil.copy()
        airfoil.point_data['wallShearStress'] = wss
        airfoil.point_data['p'] = p
        airfoil = airfoil.ptc(pass_point_data = False)

        wp_int = -airfoil.cell_data['p'][:, None]*airfoil.cell_data['Normals'][:, :2]

        wss_int = (airfoil.cell_data['wallShearStress']*airfoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        wp_int = (wp_int*airfoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)

        if compressible:
            force_p = np.array(-wp_int)
            force_v = np.array(wss_int)
        else:
            force_p = np.array(-wp_int)*self.RHO
            force_v = np.array(wss_int)*self.RHO
        force = force_p + force_v

        return force, force_p, force_v

    def force_coefficient(self, reference = False):
        """
        Compute the force coefficients for the simulation. The output is a tuple of the form `((cd, cdp, cdv), (cl, clp, clv))`,
        where `cd` is the drag coefficient, `cdp` the pressure contribution of the drag coefficient and `cdv` the viscous
        contribution of the drag coefficient. Same for the lift coefficient `cl`.

        Args:
            reference (bool, optional): If ``True``, return the force coefficients computed with the reference fields.
                If ``False``, compute the force coefficients with the fields attribute of the class. Default: ``False``   
        """
        f, fp, fv = self.force(reference = reference)

        basis = np.array([[np.cos(self.angle_of_attack), np.sin(self.angle_of_attack)], [-np.sin(self.angle_of_attack), np.cos(self.angle_of_attack)]])
        fp_rot, fv_rot = np.matmul(basis, fp), np.matmul(basis, fv)
        cp, cv = fp_rot/(.5*self.RHO*self.inlet_velocity**2), fv_rot/(.5*self.RHO*self.inlet_velocity**2)
        cdp, cdv = cp[0], cv[0]
        clp, clv = cp[1], cv[1]
        cd, cl = cdp + cdv, clp + clv

        return ((cd, cdp, cdv), (cl, clp, clv))
    
    def mean_absolute_error(self):
        """
        Compute the mean absolute error between the reference target fields and the attribute target fields of the class.
        The target fields are given in this order: velocity_x, velocity_y, pressure, kinematic turbulent viscosity.
        """
        velocity_ref = np.array(self.internal.point_data['U'][:, :2])
        pressure_ref = np.array(self.internal.point_data['p'][:, None])
        nu_t_ref = np.array(self.internal.point_data['nut'][:, None])

        absolute_error = np.abs(np.concatenate([velocity_ref - self.velocity, pressure_ref - self.pressure, nu_t_ref - self.nu_t], axis = -1))

        return absolute_error.mean(axis = 0)

    def mean_squared_error(self):
        """
        Compute the mean squared error between the reference target fields and the attribute target fields of the class.
        The target fields are given in this order: velocity_x, velocity_y, pressure, kinematic turbulent viscosity.
        """
        velocity_ref = np.array(self.internal.point_data['U'][:, :2])
        pressure_ref = np.array(self.internal.point_data['p'][:, None])
        nu_t_ref = np.array(self.internal.point_data['nut'][:, None])

        squared_error = np.concatenate([velocity_ref - self.velocity, pressure_ref - self.pressure, nu_t_ref - self.nu_t], axis = -1)**2

        return squared_error.mean(axis = 0)
    
    def r_squared(self):
        """
        Compute the r_squared between the reference target fields and the attribute target fields of the class.
        The target fields are given in this order: velocity_x, velocity_y, pressure, kinematic turbulent viscosity.
        """
        velocity_ref = np.array(self.internal.point_data['U'][:, :2])
        pressure_ref = np.array(self.internal.point_data['p'][:, None])
        nu_t_ref = np.array(self.internal.point_data['nut'][:, None])
        ref = np.concatenate([velocity_ref, pressure_ref, nu_t_ref], axis = -1)
        pred = np.concatenate([self.velocity, self.pressure, self.nu_t], axis = -1)

        return 1 - ((ref - pred)**2).sum(axis = 0)/((ref.mean(axis = 0) - pred)**2).sum(axis = 0)
    
    def coefficient_relative_error(self):
        """
        Compute the mean relative error between the reference force coefficient and the force coefficient
        computed with the attribute target fields of the class.
        The force coefficients are given in this order: drag coefficient, lift coefficient.
        """
        cd_ref, cl_ref = self.force_coefficient(reference = True)
        cd_pred, cl_pred = self.force_coefficient(reference = False)
        cd_ref, cl_ref = cd_ref[0], cl_ref[0]
        cd_pred, cl_pred = cd_pred[0], cl_pred[0]

        return np.abs((cd_ref - cd_pred)/(cd_ref + 1e-15)), np.abs((cl_ref - cl_pred)/(cl_ref + 1e-15))

    def boundary_layer(self, x, y = 0.1, extrado = True, direction = 'vertical', local_frame = False, resolution = 1000, compressible = False, reference = False):
        """
        Return the boundary layer profile or the trail profile at abscissas x over a line of length y. 
        
        The fields are returned in the following order: the position on the line (in chord length), the
        first component of the velocity in the chosen frame normalized by the inlet velocity, the
        second component of the velocity in the chosen frame normalized by the inlet velocity, the
        pressure normalized by the inlet dynamic pressure, and the turbulent kinematic viscosity
        normalized by the knimatic viscosity.

        Args:
            x (float): Abscissa in chord length. It must be strictly positive. If x < 1, return the boundary layer.
                If x >= 1, return the trail profile.
            y (float, optional): Length of the sampling line. If x < 1, this length is taken from the airfoil surface. If x >= 1,
                this length is taken from one side to the other of the trail. Default: 0.1
            extrado (bool, optional): If ``True``, the boundary layer of the extrado is returned, If ``False``, the boundary
                layer of the intrado is returned. If x>= 1, this parameter is not taken into account. Default: ``True``
            direction (str, optional): Choose between ``'vertical'`` and ``'normals'``. If x < 1, the sampling line is defined 
                as the line starting at the surface of the airfoil at abscissa x, of length y, in the direction of the normals if 
                ``'normals'`` and in the vertical direction if ``'vertical'``. If x>= 1, this parameter is not taken into account
                and the vertical direction is adopted. Default: ``'vertical'``
            local_frame (bool, optional): If ``True``, the sampled velocity components along the lines are given in the local frame,
                i.e. the frame defined by the normals. Else, the sampled velocity components are given in the cartesian frame.
                If x>= 1, this parameter is not taken into account and the cartesian frame is adopted. Default: False
                resolution (int, optional): Resolution of the sampling. Default: 1000
            compressible (bool, optional): If ``True``, add the specific mass to the normalization constant for the pressure.
                Default: ``False``
            reference (bool, optional): If ``True``, return the sampled fields of reference. If ``False``, return the sampled
                fields from the class attribute. Default: ``False``
        """
        assert x > 0, 'x must be stricly positive.'

        if x < 1:
            digits = np.array(list(map(float, self.name.split('_')[4:-1]))).astype('float64')
            camber = camber_line(digits, self.airfoil_position[:, 0])[0]
            idx_extrado = self.airfoil_position[:, 1] > camber

            if extrado:
                arg = np.argmin(np.abs(self.airfoil_position[idx_extrado, 0] - x))
                arg = np.argwhere(idx_extrado.cumsum(axis = 0) == arg)[0][0]
            else:
                arg = np.argmin(np.abs(self.airfoil_position[~idx_extrado, 0] - x))
                arg = np.argwhere((~idx_extrado).cumsum(axis = 0) == arg)[0][0]

            if direction == 'normals':
                normals = np.concatenate([self.airfoil_normals[arg], np.zeros(1)])
            
            elif direction == 'vertical':
                normals = np.array([0, 2*int(extrado) - 1, 0]).astype('float64')
            
            a, b = np.concatenate([self.airfoil_position[arg], .5*np.ones(1)]), np.concatenate([self.airfoil_position[arg], .5*np.ones(1)]) + y*normals
        else:
            c = (x - 1)*np.tan(self.angle_of_attack)
            a, b = np.array([x, c - y/2, self.internal.points[0, 2]]), np.array([x, c + y/2, self.internal.points[0, 2]])

        if reference:
            internal = self.internal.copy()
            internal.point_data['U'] = internal.point_data['U'].astype('float64')
            internal.point_data['p'] = internal.point_data['p'].astype('float64')
            internal.point_data['nut'] = internal.point_data['nut'].astype('float64')
            internal.cell_data['U'] = internal.cell_data['U'].astype('float64')
            internal.cell_data['p'] = internal.cell_data['p'].astype('float64')
            internal.cell_data['nut'] = internal.cell_data['nut'].astype('float64')
        else:
            internal = self.internal.copy()
            internal.point_data['U'] = np.concatenate([self.velocity, np.zeros_like(self.sdf)], axis = -1)
            internal.point_data['p'] = self.pressure.flatten()
            internal.point_data['nut'] = self.nu_t.flatten()
            internal = internal.ptc(pass_point_data = True)

        bl = internal.sample_over_line(a, b, resolution = resolution)
        
        if local_frame and x < 1:
            rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype('float64')
            u = (np.array(bl.point_data['U']).astype('float64')*(np.matmul(rot, normals))).sum(axis = 1)
            v = (np.array(bl.point_data['U']).astype('float64')*normals).sum(axis = 1)
        else:
            u = np.array(bl.point_data['U'][:, 0]).astype('float64')
            v = np.array(bl.point_data['U'][:, 1]).astype('float64')
        
        if compressible:
            p = np.array(bl.point_data['p']).astype('float64')/(.5*self.RHO*self.inlet_velocity**2)
        else:
            p = np.array(bl.point_data['p']).astype('float64')/(.5*self.inlet_velocity**2)
        nut = np.array(bl.point_data['nut']).astype('float64')

        if x < 1:
            yc = np.array(bl.points[:, 1]).astype('float64') - a[1]           
        else:
            yc = np.array(bl.points[:, 1]).astype('float64') - (a[1] + b[1])/2

        if x > 1:
            mask = (u != 0)
            yc, u, v, p, nut = yc[mask], u[mask], v[mask], p[mask], nut[mask]

        return yc, u/self.inlet_velocity, v/self.inlet_velocity, p, nut/self.NU

    def save(self, root):
        """
        Save the internal and the airfoil patches with the attribute targets fields of the class in the root directory.

        Args:
            root (str): Root directory where the files will be saved.
        """    
        internal = self.internal.copy()
        internal.point_data['U'] = np.concatenate([self.velocity, np.zeros_like(self.velocity[:, :1])], axis = -1)
        internal.point_data['p'] = self.pressure.flatten()
        internal.point_data['nut'] = self.nu_t.flatten()
        internal = internal.ptc(pass_point_data = True)

        airfoil = self.airfoil.copy()
        airfoil.point_data['p'] = reorganize(self.position, self.airfoil_position, self.pressure).flatten()
        airfoil.point_data['wallShearStress'] = np.concatenate([self.wallshearstress(over_airfoil = True), np.zeros(airfoil.points[:, :1].shape)], axis = -1)
        airfoil = airfoil.ptc(pass_point_data = True)

        cwd = os.getcwd()
        os.chdir('/')
        os.makedirs(osp.join(root, self.name), exist_ok = True)
        internal.save(osp.join(root, self.name, self.name + '_internal_predicted.vtu'))
        airfoil.save(osp.join(root, self.name, self.name + '_aerofoil_predicted.vtp'))
        os.chdir(cwd)