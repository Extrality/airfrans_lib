import os
import os.path as osp
import torch
import pyvista as pv

from airfrans.reorganize import reorganize
from airfrans.naca_generator import camber_line
import airfrans.sampling as sampling

class Simulation:
    """
    Wrapper to make the study of AirfRANS simulations easier. It only takes in input ``.vtu`` and ``.vtp`` files
    from the internal and airfoil patches of the simulation.
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
        self.T = torch.tensor(T) # Temperature in Kelvin

        self.reset()
    
    def reset(self):        
        self.MOL = torch.tensor(28.965338e-3) # Air molar weigth in kg/mol
        self.P_ref = torch.tensor(1.01325e5) # Pressure reference in Pa
        self.RHO = self.P_ref*self.MOL/(8.3144621*self.T) # Specific mass of air at temperature T
        self.NU = -3.400747e-6 + 3.452139e-8*self.T + 1.00881778e-10*self.T**2 - 1.363528e-14*self.T**3 # Approximation of the kinematic viscosity of air at temperature T
        self.C = 20.05*torch.sqrt(self.T) # Approximation of the sound velocity of air at temperature T        

        self.inlet_velocity = torch.tensor(float(self.name.split('_')[2])).double()
        self.angle_of_attack = torch.tensor(float(self.name.split('_')[3])*torch.pi/180).double()

        self.internal = pv.read(osp.join(self.root, self.name, self.name + '_internal.vtu'))
        self.airfoil = pv.read(osp.join(self.root, self.name, self.name + '_aerofoil.vtp'))
        self.internal = self.internal.compute_cell_sizes(length = False, volume = False)
        self.airfoil = self.airfoil.compute_cell_sizes(area = False, volume = False)

        # Input candidates
        self.surface = torch.tensor((self.internal.point_data['U'][:, 0] == 0))
        self.sdf = -torch.tensor(self.internal.point_data['implicit_distance'][:, None]).double()
        self.input_velocity = (torch.tensor([torch.cos(self.angle_of_attack), torch.sin(self.angle_of_attack)])*self.inlet_velocity).reshape(1, 2)*torch.ones_like(self.sdf)
        
        self.position = torch.tensor(self.internal.points[:, :2]).double()
        self.airfoil_position = torch.tensor(self.airfoil.points[:, :2]).double()
        self.airfoil_normals = torch.tensor(-self.airfoil.point_data['Normals'][:, :2]).double()

        self.normals = torch.zeros_like(self.input_velocity).double()
        self.normals[self.surface] = reorganize(
            self.airfoil_position,
            self.position[self.surface], 
            self.airfoil_normals
            )

        # Targets
        self.velocity = torch.tensor(self.internal.point_data['U'][:, :2]).double()
        self.pressure = torch.tensor(self.internal.point_data['p'][:, None]).double()
        self.nu_t = torch.tensor(self.internal.point_data['nut'][:, None]).double()

    def sampling_volume(self, n, density = 'uniform', targets = True):
        """
        Sample points in the internal mesh following the given density.

        Args:
            n (int): Number of sampled points.
            density (str, optional): Density from which the sampling is done. Choose between ``'uniform'`` and ``'mesh_density'``. Default: 'uniform'
            targets (bool, optional): If ``True``, velocity, pressure and kinematic turbulent viscosity will be returned in the output tensor. Default: True
        """
        if density == 'uniform': # Uniform sampling strategy
                p = torch.tensor(self.internal.cell_data['Area']/self.internal.cell_data['Area'].sum())
                sampled_cell_indices = torch.multinomial(input = p, num_samples = n, replacement = True)
        elif density == 'mesh_density': # Sample via mesh density
            sampled_cell_indices = torch.multinomial(input = torch.ones(self.internal.n_cells), num_samples = n, replacement = True)

        cell_dict = torch.tensor(self.internal.cells).reshape(-1, 5)[sampled_cell_indices, 1:]
        cell_points = self.position[cell_dict]

        if targets:
            cell_attr = torch.cat([self.sdf[cell_dict], self.velocity[cell_dict], self.pressure[cell_dict], self.nu_t[cell_dict]], dim = -1)
        else:
            cell_attr = self.sdf[cell_dict]            

        return sampling.cell_sampling_2d(cell_points, cell_attr)
    
    def sampling_surface(self, n, density = 'uniform', targets = True):
        """
        Sample points in the airfoil mesh following the given density.

        Args:
            n (int): Number of sampled points.
            density (str, optional): Density from which the sampling is done. Choose between ``'uniform'`` and ``'mesh_density'``. Default: 'uniform'
            targets (bool, optional): If ``True``, velocity, pressure and kinematic turbulent viscosity will be returned in the output tensor. Default: True
        """
        if density == 'uniform': # Uniform sampling strategy
                p = torch.tensor(self.airfoil.cell_data['Length']/self.airfoil.cell_data['Length'].sum())
                sampled_line_indices = torch.multinomial(input = p, num_samples = n, replacement = True)
        elif density == 'mesh_density': # Sample via mesh density
            sampled_line_indices = torch.multinomial(input = torch.ones(self.airfoil.n_cells), size = n, replacement = True)

        line_dict = torch.tensor(self.airfoil.lines).reshape(-1, 3)[sampled_line_indices, 1:]
        line_points = torch.tensor(self.airfoil.points)[line_dict]

        normal = torch.tensor(-self.airfoil.point_data['Normals'][line_dict, :2]) 

        if targets:
            line_attr = torch.cat([
                normal, 
                torch.tensor(self.airfoil.point_data['U'][line_dict, :2]), 
                torch.tensor(self.airfoil.point_data['p'][line_dict, None]), 
                torch.tensor(self.airfoil.point_data['nut'][line_dict, None])
                ], dim = -1)
        else:
            line_attr = normal     

        return sampling.cell_sampling_1d(line_points, line_attr)

    def sampling_mesh(self, n, targets = True):
        """
        Sample points over the simulation mesh without replacement.

        Args:
            n (int): Number of sampled points, this number has to be lower than the total number of points in the mesh.
            targets (bool, optional): If ``True``, velocity, pressure and kinematic turbulent viscosity will be returned in the output tensor. Default: True
        """
        idx = torch.arange(self.position.size(0))
        idx = torch.multinomial(input = torch.ones_like(idx, dtype = torch.float), num_samples = n)

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
            sample = torch.cat([position, surface[:, None], sdf, normals, input_velocity, velocity, pressure, nu_t], dim = -1)
        else:
            sample = torch.cat([position, surface[:, None], sdf, normals, input_velocity], dim = -1)

        return sample

    def wallshearstress(self, over_airfoil = False, reference = False):
        """
        Compute the wall shear stress.

        Args:
            over_airfoil (bool, optional): If ``True``, return the wall shear stress over the airfoil mesh. If ``False``,
            return the wall shear stress over the internal mesh. Default: False
            reference (bool, optional): If ``True``, return the wall shear stress computed with the reference velocity field.
            If ``False``, compute the wall shear stress with the velocity attribute of the class. Default: False     
        """
        if reference:
            internal = self.internal.copy()
            internal.point_data['U'] = internal.point_data['U'].astype('float64')
        else:
            internal = self.internal.copy()
            internal.point_data['U'] = torch.cat([self.velocity, torch.zeros_like(self.sdf)], dim = -1).numpy()
        jacobian = torch.tensor(internal.compute_derivative(scalars = 'U', gradient = 'jacobian').point_data['jacobian'].reshape(-1, 3, 3)[self.surface, :2, :2], dtype = torch.float64)

        s = .5*(jacobian + jacobian.transpose(1, 2))
        s = s - s.diagonal(dim1 = -2, dim2 = -1).sum(dim = -1).reshape(-1, 1, 1)*torch.eye(2)[None]/3

        wss_ = 2*self.NU.reshape(-1, 1, 1)*s
        wss_ = (wss_*self.normals[self.surface, :2].reshape(-1, 1, 2)).sum(dim = 2)

        if over_airfoil:
            wss = reorganize(self.position[self.surface], self.airfoil_position, wss_)
        else:
            wss = torch.zeros_like(self.velocity, dtype = torch.float64)
            wss[self.surface] = wss_

        return wss
    
    def force(self, compressible = False, reference = False):
        """
        Compute the force acting on the airfoil. The output is a tuple of the form ``(f, fp, fv)``, 
        where ``f`` is the force, ``fp`` the pressure contribution of the force and ``fv`` the viscous
        contribution of the force.

        Args:
            compressible (bool, optional): If ``False``, multiply the force computed with the simulation field by the specific mass. Default: False
            reference (bool, optional): If ``True``, return the force computed with the reference fields.
            If ``False``, compute the force with the fields attribute of the class. Default: False   
        """
        wss = self.wallshearstress(over_airfoil = True, reference = reference)

        if reference:
            p = torch.tensor(self.internal.point_data['p'][:, None]).double()
        else:
            p = self.pressure
        p = reorganize(self.position[self.surface], self.airfoil_position, p[self.surface])

        airfoil = self.airfoil.copy()
        airfoil.point_data['wallShearStress'] = wss.numpy()
        airfoil.point_data['p'] = p.numpy()
        airfoil = airfoil.ptc(pass_point_data = False)

        wp_int = -airfoil.cell_data['p'][:, None]*airfoil.cell_data['Normals'][:, :2]

        wss_int = (airfoil.cell_data['wallShearStress']*airfoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        wp_int = (wp_int*airfoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)

        if compressible:
            force_p = torch.tensor(-wp_int)
            force_v = torch.tensor(wss_int)
        else:
            force_p = torch.tensor(-wp_int)*self.RHO
            force_v = torch.tensor(wss_int)*self.RHO
        force = force_p + force_v

        return force, force_p, force_v

    def force_coefficient(self, reference = False):
        """
        Compute the force coefficients for the simulation. The output is a tuple of the form ``((cd, cdp, cdv), (cl, clp, clv))``,
        where ``cd`` is the drag coefficient, ``cdp`` the pressure contribution of the drag coefficient and ``cdv`` the viscous
        contribution of the drag coefficient. Same for the lift coefficient ``cl``.

        Args:
            reference (bool, optional): If ``True``, return the force coefficients computed with the reference fields.
            If ``False``, compute the force coefficients with the fields attribute of the class. Default: False   
        """
        f, fp, fv = self.force(reference = reference)

        basis = torch.tensor([[torch.cos(self.angle_of_attack), torch.sin(self.angle_of_attack)], [-torch.sin(self.angle_of_attack), torch.cos(self.angle_of_attack)]])
        fp_rot, fv_rot = torch.matmul(basis, fp), torch.matmul(basis, fv)
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
        velocity_ref = torch.tensor(self.internal.point_data['U'][:, :2])
        pressure_ref = torch.tensor(self.internal.point_data['p'][:, None])
        nu_t_ref = torch.tensor(self.internal.point_data['nut'][:, None])

        absolute_error = torch.abs(torch.cat([velocity_ref - self.velocity, pressure_ref - self.pressure, nu_t_ref - self.nu_t], dim = -1))

        return absolute_error.mean(dim = 0)

    def mean_squared_error(self):
        """
        Compute the mean squared error between the reference target fields and the attribute target fields of the class.
        The target fields are given in this order: velocity_x, velocity_y, pressure, kinematic turbulent viscosity.
        """
        velocity_ref = torch.tensor(self.internal.point_data['U'][:, :2])
        pressure_ref = torch.tensor(self.internal.point_data['p'][:, None])
        nu_t_ref = torch.tensor(self.internal.point_data['nut'][:, None])

        squared_error = torch.cat([velocity_ref - self.velocity, pressure_ref - self.pressure, nu_t_ref - self.nu_t], dim = -1)**2

        return squared_error.mean(dim = 0)
    
    def r_squared(self):
        """
        Compute the r_squared between the reference target fields and the attribute target fields of the class.
        The target fields are given in this order: velocity_x, velocity_y, pressure, kinematic turbulent viscosity.
        """
        velocity_ref = torch.tensor(self.internal.point_data['U'][:, :2])
        pressure_ref = torch.tensor(self.internal.point_data['p'][:, None])
        nu_t_ref = torch.tensor(self.internal.point_data['nut'][:, None])
        ref = torch.cat([velocity_ref, pressure_ref, nu_t_ref], dim = -1)
        pred = torch.cat([self.velocity, self.pressure, self.nu_t], dim = -1)

        return 1 - ((ref - pred)**2).sum(dim = 0)/((ref.mean(dim = 0, keepdim = True) - pred)**2).sum(dim = 0)
    
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

        return torch.abs((cd_ref - cd_pred)/(cd_ref + 1e-15)), torch.abs((cl_ref - cl_pred)/(cl_ref + 1e-15))

    def boundary_layer(self, x, y = 0.1, extrado = True, direction = 'vertical', local_frame = False, resolution = 1000, compressible = False, reference = False):
        """
        Return the boundary layer profile or the trail profile at abscissas x over a line of length y. The fields
        returned are returned in the following order: the position on the line (in chord length), the
        first component of the velocity in the chosen frame normalized by the inlet velocity, the
        second component of the velocity in the chosen frame normalized by the inlet velocity, the
        pressure normalized by the inlet dynamic pressure, and the turbulent kinematic viscosity
        normalized by the knimatic viscosity.

        Args:
            x (torch.tensor): Abscissa in chord length. It must be strictly positive. If x < 1, return the boundary layer.
            If x >= 1, return the trail profile.
            y (float, optional): Length of the sampling line. If x < 1, this length is taken from the airfoil surface. If x >= 1,
            this length is taken from one side to another of the trail. Default: 0.1
            extrado (bool, optional): If ``True``, the boundary layer of the extrado is returned, If ``False``, the boundary
            layer of the intrado is returned. If x>= 1, this parameter is not taken into account.
            direction (str, optional): Choose between ``'vertical'`` and ``'normals'``. If x < 1, the sampling line is defined 
            as the line starting at the surface of the airfoil at abscissa x, of length y, in the direction of the normals if 
            ``'normals'`` and in the vertical direction if ``'vertical'``. If x>= 1, this parameter is not taken into account
            and the vertical direction is adopted. Default: 'vertical'
            local_frame (bool, optional): If ``True``, the sampled velocity components along the lines are given in the local frame,
            i.e. the frame defined by the normals. Else, the sampled velocity components are given in the cartesian frame.
            If x>= 1, this parameter is not taken into account and the cartesian frame is adopted. Default: False
            resolution (int, optional): Resolution of the sampling. Default: 1000
            compressible (bool, optional): If ``True``, add the specific mass to the normalization constant for the pressure.
            reference (bool, optional): If ``True``, return the sampled fields of reference. If ``False``, return the sampled
            fields from the class attribute. Default: False
        """
        assert x > 0, 'x must be stricly positive.'

        if x < 1:
            digits = torch.tensor(list(map(float, self.name.split('_')[4:-1]))).double()
            camber = camber_line(digits, self.airfoil_position[:, 0])[0]
            idx_extrado = self.airfoil_position[:, 1] > camber

            if extrado:
                arg = torch.argmin(torch.abs(self.airfoil_position[idx_extrado, 0] - x))
                arg = torch.argwhere(idx_extrado.cumsum(dim = 0) == arg)[0][0]
            else:
                arg = torch.argmin(torch.abs(self.airfoil_position[~idx_extrado, 0] - x))
                arg = torch.argwhere((~idx_extrado).cumsum(dim = 0) == arg)[0][0]

            if direction == 'normals':
                normals = torch.cat([self.airfoil_normals[arg], torch.zeros(1)])
            
            elif direction == 'vertical':
                normals = torch.tensor([0, 2*int(extrado) - 1, 0]).double()
            
            a, b = torch.cat([self.airfoil_position[arg], .5*torch.ones(1)]), torch.cat([self.airfoil_position[arg], .5*torch.ones(1)]) + y*normals
        else:
            c = (x - 1)*torch.tan(self.angle_of_attack)
            a, b = torch.tensor([x, c - y/2, self.internal.points[0, 2]]), torch.tensor([x, c + y/2, self.internal.points[0, 2]])

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
            internal.point_data['U'] = torch.cat([self.velocity, torch.zeros_like(self.sdf)], dim = -1).numpy()
            internal.point_data['p'] = self.pressure.flatten().numpy()
            internal.point_data['nut'] = self.nu_t.flatten().numpy()
            internal = internal.ptc(pass_point_data = True)

        bl = internal.sample_over_line(a.numpy(), b.numpy(), resolution = resolution)
        
        if local_frame and x < 1:
            rot = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).double()
            u = (torch.tensor(bl.point_data['U']).double()*(torch.matmul(rot, normals))).sum(dim = 1)
            v = (torch.tensor(bl.point_data['U']).double()*normals).sum(dim = 1)
        else:
            u = bl.point_data['U'][:, 0]
            v = bl.point_data['U'][:, 1]
        
        if compressible:
            p = torch.tensor(bl.point_data['p']).double()/(.5*self.RHO*self.inlet_velocity**2)
        else:
            p = torch.tensor(bl.point_data['p']).double()/(.5*self.inlet_velocity**2)
        nut = torch.tensor(bl.point_data['nut']).double()

        if x < 1:
            yc = torch.tensor(bl.points[:, 1]).double() - a[1]           
        else:
            yc = torch.tensor(bl.points[:, 1]).double() - (a[1] + b[1])/2

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
        internal.point_data['U'] = torch.cat([self.velocity, torch.zeros_like(self.velocity[:, :1])], dim = -1).numpy()
        internal.point_data['p'] = self.pressure.flatten().numpy()
        internal.point_data['nut'] = self.nu_t.flatten().numpy()
        internal = internal.ptc(pass_point_data = True)

        airfoil = self.airfoil.copy()
        airfoil.point_data['p'] = reorganize(self.position, self.airfoil_position, self.pressure).flatten().numpy()
        airfoil.point_data['wallShearStress'] = torch.cat([self.wallshearstress(over_airfoil = True), torch.zeros(airfoil.points[:, :1].shape)], dim = -1).numpy()
        airfoil = airfoil.ptc(pass_point_data = True)

        cwd = os.getcwd()
        os.chdir('/')
        os.makedirs(osp.join(root, self.name), exist_ok = True)
        internal.save(osp.join(root, self.name, self.name + '_internal_predicted.vtu'))
        airfoil.save(osp.join(root, self.name, self.name + '_aerofoil_predicted.vtp'))
        os.chdir(cwd)