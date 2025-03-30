column_headers = [
    "Entry",
    "Longitude",       # The geographical longitude of the vehicle.
    "Latitude",        # The geographical latitude of the vehicle.
    "Altitude",        # The altitude of the vehicle.
    "Heading",         # The direction in which the vehicle is traveling (in degrees).
    "Speed",           # The vehicle's speed (e.g., in meters per second or kilometers per hour).
    "Acceleration",    # The acceleration of the vehicle (in meters per second squared).
    "Vehicle Length",  # The physical length of the vehicle (in meters).
    "Vehicle Width",   # The physical width of the vehicle (in meters).
    "Vehicle Count",   # A computed feature indicating how many vehicles sent messages during that second.
    "Entity",
    "Collision"
]

input_dim = 9