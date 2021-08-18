import numpy as np

stability_val = {'Very unstable': 0, 'Moderate unstable': 1, 'Near neutral': 2, 'Moderate stable': 3, 'Very stable': 4}

lateral_intensity = [0.5, 0.3, 0.2, 0.15, 0.1]


def get_odour_conc_landscape(x, y, radius, centre, shape='Volcano', k=10.0, tau=0.1):
    if shape == 'bar':
        if not centre[0]:
            c = k * np.exp(-(y - centre[1])**2/(2.0*radius**2)) 
        else:
            c = k * np.exp(-(x - centre[0])**2/(2.0*radius**2))
    else:
        dis = np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
        if dis >= radius/2.0:
            c = k * np.exp(tau*(radius/2.0 - dis))
        else:
            if shape == 'Volcano':
                c = k * np.exp(tau*(dis - radius/2.0))
            elif shape == 'Well':
                c = 0.0
            elif shape == 'Mesa':
                c = k
            elif shape == 'Linear':
                c = k - 0.2 * (dis - radius/2.0)
            else:
                c = 0.0
    return c


def get_odour_concentration(q, u, w_dir, x, y, sx=0, sy=0, stability='Near neutral'):
    # if it is at the source
    if (x == sx) and (y == sy):
        return get_odour_concentration(q, u, w_dir, sx+0.01, sy, sx, sy, stability)
    x1 = x - sx
    y1 = y - sy

    ux = u * np.cos(w_dir)
    uy = u * np.sin(w_dir)

    # distance from source
    dis = np.sqrt(x1**2 + y1**2)

    # angle between the point and the wind to judge the down-wind area
    angle = np.arccos((x1 * ux + y1 * uy)/(dis * np.sqrt(ux**2+uy**2)))

    # add 1e-15 to avoid divided by zero warning
    sig_xy = dis * lateral_intensity[stability_val[stability]] + 1e-15
    # distance from source and projected to the wind direction
    dis_projected = dis*np.sin(angle)

    c = q / (u * sig_xy * np.sqrt(2. * np.pi)) * np.exp(-dis_projected ** 2 / (2.0 * sig_xy ** 2))

    return c*(np.cos(angle) > 0)

class OdourField:
    def __init__(self, q, u, w_dir, radius=5.0, sx=0, sy=0, stability='Near neutral', shape='Volcano'):
        self.release_rate = q
        self.wind_speed = u
        self.odour_landscape = shape
        self.odour_landscape_radius = radius

        self.wind_direction = w_dir
        self.source_pos = [sx, sy]
        self.stability = stability

    def get_odour_concentration(self, x, y):
        if self.wind_speed == 0:
            c = get_odour_conc_landscape(x, y, self.odour_landscape_radius, self.source_pos, 
                                         shape=self.odour_landscape, k=10.0, tau=0.1)
        else:
            x1 = x - self.source_pos[0]
            y1 = y - self.source_pos[1]

            ux = self.wind_speed * np.cos(self.wind_direction)
            uy = self.wind_speed * np.sin(self.wind_direction)

            # distance from source
            dis = np.sqrt(x1 ** 2 + y1 ** 2)

            # angle between the point and the wind to judge the down-wind area
            angle = np.arccos((x1 * ux + y1 * uy) / (dis * np.sqrt(ux ** 2 + uy ** 2)))

            # add 1e-15 to avoid divided by zero warning
            sig_xy = dis * lateral_intensity[stability_val[self.stability]] + 1e-15
            # distance from source and projected to the wind direction
            dis_projected = dis * np.sin(angle)

            c = self.release_rate / (self.wind_speed * sig_xy * np.sqrt(2. * np.pi)) * np.exp(-dis_projected ** 2 / (2.0 * sig_xy ** 2))

            c = c * (np.cos(angle) > 0)

        return c
