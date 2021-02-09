import numpy as np



def TSM(**kwargs):

    """

    Simplistic thermodynamic zero-layer sea ice model for the calculation of sea ice thickness

    Parameters
    ----------
    years : integer
            length of the simulation in years

    delflux : float
            Additional incoming heatflux [W/m^2] which is added to the incoming longwave radiation

    h_ice0 : float
            initial sea ice thickness [m]


    Returns
    -------
    h_ice : ndarray
            array of sea ice thickness [m] for each day of the simulation

    h_snow : ndarray
            array of snow thickness [m] for each day of the simulation

    T_mxl : ndarray
            array of the temperature of the ocean mixed layer [°C] for aech day of the simulation
    """

    # get kwargs and set defaults
    years = kwargs.get('years', 10)

    # Difference of other heatflux to reach 1m sea ice thickness in summer
    delflux = kwargs.get('delflux', np.ones(years * 365)*61.7)

    # Initial sea ice thickness
    h_ice0 = kwargs.get('h_ice0', 1.12)

    days = years * 365  # duration of simulation in days

    # Initialize arrays
    h_ice = np.zeros(days)
    h_snow = np.zeros(days)


    # set parameters
    T_bot = 273.15 - 1.8  # Temperature at the bottom [C]
    h_ice[0] = h_ice0  # Initial Sea ice thickness calculated from MPI jan mean [m]
    h_snow[0] = 0.317  # Initial snow thickness [m]
    T_mxl = np.ones(days) * T_bot  # Initial mixed layer temp
    Q_oce = 5  # Heat flux from the ocean [W/m^2]
    hocean = 50  # Depth of the mixed layer [m]


    # set physical constants
    L = 334000  # Latent heat of freezing for water [ J/kg ]
    rho_ice = 970  # density of ice [ kg/m^ 3 ]
    rho_snow = 330  # density of snow [kg/m^3]
    k_ice = 2.2  # heat conductivity of ice [W/(m K) ]
    k_snow = 0.3  # heat conductivity of snow [W/(m K) ]
    sec_per_day = 86400  # How many seconds in one day
    c_water = 4000  # Heat capacity of water [J/kg K]
    albedo_water = 0.1  # Albedo of water
    rho_w = 1025  # Desity of water [kg/m^3]
    eps_sigma = 0.95 * 5.67e-8  # Constant in Boltzman-law

    # Define functions

    # Short wave radiation from the sun
    def shortwave(day):
        doy = np.mod(day, 365)
        return 314. * np.exp(-(doy - 164) ** 2 / 4608.)

    # sensible + latent + longwave
    def otherfluxes(day):
        doy = np.mod(day, 365)
        return 118. * np.exp(-0.5 * (doy - 206.) ** 2 / (53 ** 2)) + 179. - delflux[day]

    # Albedo of ice
    def albedo(day):
        doy = np.mod(day, 365)
        return -0.431 / (1 + ((doy - 207) / 44.5) ** 2) + 0.914

    # Temperature of the ocean mixed layer
    def temp_mxl(doy, T_mxl_old):
        Q_in = (1 - albedo_water) * shortwave(doy) + otherfluxes(doy)
        Q_out = eps_sigma * T_mxl_old ** 4  # approximation of outgoing LW flux with old mxl Temp
        T_mxl = T_mxl_old + (Q_in - Q_out) / c_water * (1 / (rho_w * hocean)) * sec_per_day

        return T_mxl

    # Temperature of the ice surface
    def surftemp(day, h_ice, h_snow):
        a = eps_sigma
        b = 0
        c = 0
        d = 1 / (h_ice / k_ice + h_snow / k_snow)
        e = -(1 - albedo(day)) * shortwave(day) - otherfluxes(day) - T_bot / (h_ice / k_ice + h_snow / k_snow)
        t_tempor = np.roots([a, b, c, d, e])
        t_surf = np.real(max(t_tempor[np.isreal(t_tempor)]))

        return t_surf

    # Increase in snow thickness per day
    def snowfall(day):
        doy = np.mod(day, 365)
        if doy > 232 and doy <= 304:
            return 0.30 / 72.
        elif doy > 304 or doy < 122:
            return 0.05 / 176.
        elif doy >= 122 and doy < 154:
            return 0.05 / 31.
        else:
            return 0

    # Integrating function
    def sea_ice_snow(day, h_ice, h_snow):
        # Avoid Q_ice = inf
        if h_ice == 0:
            h_ice = 0.01

        # Check if surface temperature is below freezing
        T_surf = surftemp(day, h_ice, h_snow)
        if T_surf > 273.15:
            T_surf = 273.15

        # Calculate ice growth at the bottom of the sea ice
        Q_ice = -(T_surf - T_bot) / ((h_ice / k_ice) + (h_snow / k_snow))
        dh_dt_bot = 1 / (rho_ice * L) * (Q_ice - Q_oce)

        # Calculate eventual melting at the ice surface
        Q_surf_out = eps_sigma * T_surf ** 4
        Q_surf_in = (1 - albedo(day)) * shortwave(day) + otherfluxes(day)
        Q_surf_net = Q_surf_out - Q_surf_in - Q_ice
        dh_dt_top = Q_surf_net / (rho_ice * L)

        # Add sea ice growth from surface and bottom
        dh_ice = (dh_dt_bot + dh_dt_top) * sec_per_day

        # Calculate snow change
        dh_snow = snowfall(day) + dh_dt_top * sec_per_day

        return dh_ice, dh_snow

    # Calculate Sea ice
    for day in np.arange(0, days-1):

        if T_mxl[day] > T_bot:
            T_mxl[day+1] = temp_mxl(day, T_mxl[day])
            h_ice[day+1] = 0
        else:
            change = sea_ice_snow(day, h_ice[day], h_snow[day])
            h_ice[day+1] = h_ice[day] + change[0]
            h_snow[day+1] = h_snow[day] + change[1]

            if h_snow[day+1] < 0:
                h_snow[day+1] = 0

            if h_ice[day+1] < 0:
                h_ice[day+1] = 0
                T_mxl[day+1] = temp_mxl(day, T_mxl[day])

    return h_ice, h_snow, T_mxl - 273.15

