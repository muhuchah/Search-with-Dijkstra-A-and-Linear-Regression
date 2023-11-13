# Import necessary libraries
import numpy as np
import pandas as pd
from heapq import heappush, heappop
import time
import math

# Read data from the CSV file into a Pandas DataFrame
df = pd.read_csv("~/Downloads/Dataset.csv")

class airport_information:
    def __init__(self, city, country, latitude, longitude, altitude):
        self.city = city
        self.country = country
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

# Read information form dataset and save in datastructures
airports = set()
airports_name_to_index = dict()
airports_index_to_name = np.empty(4000, dtype=object)
airports_information = np.empty(4000, dtype=airport_information)

ind = 0
for i in df.index:
    airport = df["SourceAirport"][i]
    if airport not in airports:
        airports.add(airport)
        if df["SourceAirport_Country"][i] == "Dubai":
            print(airport)
        airports_name_to_index[airport] = ind
        airports_index_to_name[ind] = airport
        inf = airport_information(df["SourceAirport_City"][i], df["SourceAirport_Country"][i],
                                   df["SourceAirport_Latitude"][i], df["SourceAirport_Longitude"][i],
                                   df["SourceAirport_Altitude"][i])
        airports_information[ind] = inf
        ind += 1
        
    airport = df["DestinationAirport"][i]
    if airport not in airports:
        airports.add(airport)
        airports_name_to_index[airport] = ind
        airports_index_to_name[ind] = airport
        inf = airport_information(df["DestinationAirport_City"][i], df["DestinationAirport_Country"][i],
                                   df["DestinationAirport_Latitude"][i], df["DestinationAirport_Longitude"][i],
                                   df["DestinationAirport_Altitude"][i])
        airports_information[ind] = inf
        ind += 1

number_of_airports = len(airports)

# calculate cost of flights
def calc(distance, flytime, price):
    result = (1 * distance) + (80 * flytime) + (1 * price)
    return result

# information of flights
class flight_information:
    def __init__(self, source_airport, destination_airport, airline, distance, flytime, price):
        self.source_airport = source_airport
        self.destination_airport = destination_airport
        self.airline = airline
        self.distance = distance 
        self.flytime = flytime
        self.price = price
        self.cost = calc(distance, flytime, price)
        
class flights_list:
    def __init__(self):
        self.flights = []
    def add_flight(self, flight):
        self.flights.append(flight)

class ghf:
    def __init__(self, g, h):
        self.g = g
        self.h = h
        self.f = self.g + self.h
    
    def set_g(self, g):
        self.g = g
        self.f = self.g + self.h

    def set_h(self, h):
        self.h = h
        self.f = self.g + self.h

# create Graph
flights_information = np.empty(4000, dtype=flights_list)
for i in range(number_of_airports):
    flights_information[i] = flights_list()

for i in df.index:
    index_source_airport = airports_name_to_index[df["SourceAirport"][i]]
    index_destination_airport = airports_name_to_index[df["DestinationAirport"][i]]
    airline = df["Airline"][i]
    distance = df["Distance"][i]
    flytime = df["FlyTime"][i]
    price = df["Price"][i]
    if distance < 0 or flytime < 0 or price < 0:
        continue
    inf = flight_information(index_source_airport, index_destination_airport, airline, distance, flytime, price)
    flights_information[index_source_airport].add_flight(inf)

# Get input
inp = input()
#inp = "Imam Khomeini International Airport - Raleigh Durham International Airport"
source_airport, destination_airport = inp.split('-')
source_airport = source_airport.strip()
destination_airport = destination_airport.strip()
index_source_airport = airports_name_to_index[source_airport]
index_destination_airport = airports_name_to_index[destination_airport]

# calculate heuristic
def heuristic(cur_airport, destination):
    # Euclidean
    x = math.fabs (airports_information[cur_airport].latitude - airports_information[destination].latitude)
    y = math.fabs (airports_information[cur_airport].longitude - airports_information[destination].longitude)
    z = math.fabs (airports_information[cur_airport].altitude - airports_information[destination].altitude)

    return ((x*x) + (y*y) + (z*z)) ** 0.5

# A* Algorithm
start_time = time.time()

openList = []
closedList = np.empty(4000, dtype=bool)
closedList.fill(False)
costs = np.empty(4000, dtype=ghf)
for i in range(4000):
    costs[i] = ghf(math.inf, math.inf)
path = np.empty(4000, dtype=int)
path.fill(-1)
cost_detail = np.empty(4000, dtype=flight_information)
cost_inf = flight_information(-1, -1, -1, -1, -1, -1)
cost_detail.fill(cost_inf)

costs[index_source_airport].set_g(0)
for airport in range(number_of_airports):
    costs[airport].set_h(heuristic(airport, index_destination_airport))

heappush (openList, (costs[index_source_airport].f, index_source_airport))

while openList:
    # Get and remove the node with largest f
    cost, cur_airport = heappop(openList)
    closedList[cur_airport] = True
    if cur_airport == index_destination_airport:
        break
    if cost != costs[cur_airport].f:
        continue
    for flight in flights_information[cur_airport].flights:
        if not closedList[flight.destination_airport]:
            destination_g = costs[cur_airport].g + flight.cost
            destination_h = costs[flight.destination_airport].h
            destination_f = destination_g + destination_h
            
            if costs[flight.destination_airport].f > destination_f:
                costs[flight.destination_airport].set_g(destination_g)
                heappush(openList, (destination_f, flight.destination_airport))
                path[flight.destination_airport] = cur_airport
                cost_detail[flight.destination_airport] = flight

total_time = round(time.time() - start_time, 6)

# save path in list way
destination = index_destination_airport
way = []
total_distance = 0
total_flytime = 0
total_price = 0
r = 2 #round_value

while destination != -1:
    way.append(cost_detail[destination])
    destination = path[destination]
way.pop()
way = way[::-1]

# create file for A*
f = open("1-UIAI4021-PR1-Q1(A*).txt", "w")
f.write("A* Algorithm\n")
f.write(f"Execution Time: {round(total_time, 6)} Seconds\n")
f.write(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-\n")
if len(way):
    for index, flight in enumerate(way):
        f.write(f"Flight #{index+1} ({flight.airline}):\n")
        f.write(f"From: {airports_index_to_name[flight.source_airport]} - {airports_information[flight.source_airport].city}, {airports_information[flight.source_airport].country}\n")
        f.write(f"To: {airports_index_to_name[flight.destination_airport]} - {airports_information[flight.destination_airport].city}, {airports_information[flight.destination_airport].country}\n")
        f.write(f"Duration: {round(flight.distance, r)} km\n")
        f.write(f"Time: {round(flight.flytime, r)} h\n")
        f.write(f"Price: {round(flight.price, r)} $\n")
        f.write("----------------------------\n")
        
    f.write(f"Total Price: {round(total_price, r)} $\n")
    f.write(f"Total Duration: {round(total_distance, r)} km\n")
    f.write(f"Total Time: {round(total_flytime, r)} h\n")
else:
    f.write("Way Not Found!\n")
f.close()

# Dijkstar Algorithm
start_time = time.time()

costs = np.empty(4000, dtype=float)
costs.fill(math.inf)
path = np.empty(4000, dtype=int)
path.fill(-1)
cost_detail = np.empty(4000, dtype=flight_information)
cost_inf = flight_information(-1, -1, -1, -1, -1, -1)
cost_detail.fill(cost_inf)

open_list = []
heappush(open_list, (0, index_source_airport))
costs[index_source_airport] = 0

while open_list:
    cost, airport = heappop(open_list)
    if cost != costs[airport]: 
        continue
    for flight in flights_information[airport].flights:
        if flight.cost + cost < costs[flight.destination_airport]:
            costs[flight.destination_airport] = flight.cost + cost
            heappush(open_list, (costs[flight.destination_airport], flight.destination_airport))
            path[flight.destination_airport] = airport
            cost_detail[flight.destination_airport] = flight

total_time = round(time.time() - start_time, 6)

# save path in list way
destination = index_destination_airport
way = []
total_distance = 0
total_flytime = 0
total_price = 0
r = 2 #round_value

while destination != -1:
    way.append(cost_detail[destination])
    destination = path[destination]
way.pop()
way = way[::-1]

# create file for Dijkstar
f = open("1-UIAI4021-PR1-Q1(Dijkstra).txt", "w")
f.write("Dijkstra Algorithm\n")
f.write(f"Execution Time: {round(total_time, 6)} Seconds\n")
f.write(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-\n")
if len(way):
    for index, flight in enumerate(way):
        f.write(f"Flight #{index+1} ({flight.airline}):\n")
        f.write(f"From: {airports_index_to_name[flight.source_airport]} - {airports_information[flight.source_airport].city}, {airports_information[flight.source_airport].country}\n")
        f.write(f"To: {airports_index_to_name[flight.destination_airport]} - {airports_information[flight.destination_airport].city}, {airports_information[flight.destination_airport].country}\n")
        f.write(f"Duration: {round(flight.distance, r)} km\n")
        f.write(f"Time: {round(flight.flytime, r)} h\n")
        f.write(f"Price: {round(flight.price, r)} $\n")
        f.write("----------------------------\n")
        
    f.write(f"Total Price: {round(total_price, r)} $\n")
    f.write(f"Total Duration: {round(total_distance, r)} km\n")
    f.write(f"Total Time: {round(total_flytime, r)} h\n")
else:
    f.write("Way Not Found!\n")
f.close()

