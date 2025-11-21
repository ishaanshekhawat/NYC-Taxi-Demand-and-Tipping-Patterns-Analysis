# NYC-Taxi-Demand-and-Tipping-Patterns-Analysis

**About the Data** - This dataset is published monthly by the NYC Taxi & Limousine Commission (TLC) and contains millions of taxi trips.

**About the Schema**

**1. VendorID**

ID of the company that provided the taxi.
Common values:
1 = Creative Mobile Technologies (CMT)
2 = VeriFone (VTS)

**2. tpep_pickup_datetime**

Timestamp (date + time) when the passenger got into the taxi.

**3. tpep_dropoff_datetime**

Timestamp when the passenger got out of the taxi.

**4. passenger_count**

Number of passengers (as entered by the driver).
Not always accurate (manual input).

**5. trip_distance**

Distance of the trip in miles.
Calculated by the taxi’s meter (GPS-based).

**6. RatecodeID**

Type of rate used for the trip.
Common values:
1 = Standard rate
2 = JFK
3 = Newark
4 = Nassau/Westchester
5 = Negotiated fare
6 = Group ride

**7. store_and_fwd_flag**

Whether the trip record was stored in the vehicle before being sent to the server.
Values:
Y = stored (no network during trip)
N = sent in real-time

**8. PULocationID
9. DOLocationID**

These are geographical zone IDs defined by TLC.
Map to neighborhoods (e.g., Midtown, Queens, JFK Airport).
Need to join them with a lookup file (taxi_zone_lookup.csv).

**10. payment_type**

How the rider paid:
1	Credit card
2	Cash
3	No charge
4	Dispute
5	Unknown
6	Voided trip

**11. fare_amount**

Base fare for the trip (distance × time).

**12. extra**

Extra charges:
1 AM–6 AM night surcharge
Peak hour weekday surcharge (4–8 PM)

**13. mta_tax**

Flat $0.50 tax for all rides in NYC.

**14. tip_amount**

Tip paid by the passenger.
Typically via credit card.
Cash tips are not included.

**15. tolls_amount**

Tolls paid (e.g., bridges, tunnels).

**16. improvement_surcharge**

Fixed $0.30 added to every trip.

**17. total_amount**

Final amount charged to the passenger (including fare + extras + tip + tolls).

**18. congestion_surcharge**

Congestion fees for rides entering certain zones:
$2.50 Yellow Taxi
$2.75 Green Taxi
Applies in Manhattan south of 96th St.

**19. Airport_fee**

$1.25 fee when dropping or picking up at LaGuardia or JFK.

**20. cbd_congestion_fee**

Newer congestion charge (post-2023 rollout), similar to congestion_surcharge but more updated version based on zone/time.
