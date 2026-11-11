#!/usr/bin/env python3
"""
recon10s_nexradsites -- NEXRAD (WSR-88D) sites, embedded so lookups need no network.

Carried over from MWSynth 0.169 (nexrad_stations.py, unused there): same
source, same parser, same API. The table is regenerated from NCEI's station list
(https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt) using the
file's own fixed-width columns (from its dash-underline header) and keeping
STNTYPE == "NEXRAD" only: the 47 TDWR sites in the same list are a different
network and are not in the unidata-nexrad-level2 archive. 163 sites, including
the tropical ones outside CONUS (Puerto Rico, Guam, Hawaii, Alaska, Korea,
Japan, Azores). Added here: the site elevation, for beam geometry.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple


class RadarStation(NamedTuple):
    icao: str
    name: str
    state: str
    lat: float
    lon: float
    elev_ft: float = 0.0


_RAW_STATIONS = [
    ('KABR', 'ABERDEEN', 'SD', 45.455833, -98.413333, 1383.0),
    ('KABX', 'ALBUQUERQUE', 'NM', 35.149722, -106.82388, 5951.0),
    ('KAKQ', 'NORFOLK RICH', 'VA', 36.98405, -77.007361, 255.0),
    ('KAMA', 'AMARILLO', 'TX', 35.233333, -101.70927, 3703.0),
    ('KAMX', 'MIAMI', 'FL', 25.611083, -80.412667, 111.0),
    ('KAPX', 'GAYLORD', 'MI', 44.90635, -84.719533, 1561.0),
    ('KARX', 'LA CROSSE', 'WI', 43.822778, -91.191111, 1357.0),
    ('KATX', 'SEATTLE', 'WA', 48.194611, -122.49569, 642.0),
    ('KBBX', 'BEALE AFB', 'CA', 39.495639, -121.63161, 221.0),
    ('KBGM', 'BINGHAMTON', 'NY', 42.199694, -75.984722, 1703.0),
    ('KBHX', 'EUREKA', 'CA', 40.498583, -124.29216, 2516.0),
    ('KBIS', 'BISMARCK', 'ND', 46.770833, -100.76055, 1755.0),
    ('KBLX', 'BILLINGS', 'MT', 45.853778, -108.6068, 3703.0),
    ('KBMX', 'BIRMINGHAM', 'AL', 33.172417, -86.770167, 759.0),
    ('KBOX', 'BOSTON', 'MA', 41.955778, -71.136861, 232.0),
    ('KBRO', 'BROWNSVILLE', 'TX', 25.916, -97.418967, 88.0),
    ('KBUF', 'BUFFALO', 'NY', 42.948789, -78.736781, 790.0),
    ('KBYX', 'KEY WEST', 'FL', 24.5975, -81.703167, 89.0),
    ('KCAE', 'COLUMBIA', 'SC', 33.948722, -81.118278, 345.0),
    ('KCBW', 'HOULTON', 'ME', 46.03925, -67.806431, 860.0),
    ('KCBX', 'BOISE', 'ID', 43.490217, -116.23603, 3172.0),
    ('KCCX', 'STATE COLLEGE', 'PA', 40.923167, -78.003722, 2486.0),
    ('KCLE', 'CLEVELAND', 'OH', 41.413217, -81.859867, 860.0),
    ('KCLX', 'CHARLESTON', 'SC', 32.655528, -81.042194, 229.0),
    ('KCRI', 'ROC FAA REDUNDANT RDA 1', 'OK', 35.238333, -97.46, 1315.0),
    ('KCRP', 'CORPUS CHRISTI', 'TX', 27.784017, -97.51125, 142.0),
    ('KCXX', 'BURLINGTON', 'VT', 44.511, -73.166431, 431.0),
    ('KCYS', 'CHEYENNE', 'WY', 41.151919, -104.80603, 6193.0),
    ('KDAX', 'SACRAMENTO', 'CA', 38.501111, -121.67783, 144.0),
    ('KDDC', 'DODGE CITY', 'KS', 37.760833, -99.968889, 2671.0),
    ('KDFX', 'LAUGHLIN AFB', 'TX', 29.273139, -100.28033, 1196.0),
    ('KDGX', 'JACKSON BRANDON', 'MS', 32.279944, -89.984444, 609.0),
    ('KDIX', 'PHILADELPHIA', 'NJ', 39.947089, -74.410731, 230.0),
    ('KDLH', 'DULUTH', 'MN', 46.836944, -92.209722, 1542.0),
    ('KDMX', 'DES MOINES', 'IA', 41.7312, -93.722869, 1095.0),
    ('KDOX', 'DOVER AFB', 'DE', 38.825767, -75.440117, 164.0),
    ('KDTX', 'DETROIT', 'MI', 42.7, -83.471667, 1216.0),
    ('KDVN', 'DAVENPORT', 'IA', 41.611667, -90.580833, 851.0),
    ('KDYX', 'DYESS AFB', 'TX', 32.5385, -99.254333, 1582.0),
    ('KEAX', 'KANSAS CITY', 'MO', 38.81025, -94.264472, 1092.0),
    ('KEMX', 'TUCSON', 'AZ', 31.89365, -110.63025, 5319.0),
    ('KENX', 'ALBANY', 'NY', 42.586556, -74.064083, 1935.0),
    ('KEOX', 'FORT RUCKER', 'AL', 31.460556, -85.459389, 537.0),
    ('KEPZ', 'EL PASO', 'NM', 31.873056, -106.698, 4218.0),
    ('KESX', 'LAS VEGAS', 'NV', 35.70135, -114.89165, 4948.0),
    ('KEVX', 'EGLIN AFB', 'FL', 30.565033, -85.921667, 221.0),
    ('KEWX', 'AUSTIN SAN ANTONIO', 'TX', 29.704056, -98.028611, 767.0),
    ('KEYX', 'EDWARDS', 'CA', 35.09785, -117.56075, 2873.0),
    ('KFCX', 'ROANOKE', 'VA', 37.0244, -80.273969, 2965.0),
    ('KFDR', 'ALTUS AFB', 'OK', 34.362194, -98.976667, 1315.0),
    ('KFDX', 'CANNON AFB', 'NM', 34.634167, -103.61888, 4698.0),
    ('KFFC', 'ATLANTA', 'GA', 33.36355, -84.56595, 972.0),
    ('KFSD', 'SIOUX FALLS', 'SD', 43.587778, -96.729444, 1495.0),
    ('KFSX', 'FLAGSTAFF', 'AZ', 34.574333, -111.19844, 7514.0),
    ('KFTG', 'DENVER FRONT RANGE AP', 'CO', 39.786639, -104.5458, 5611.0),
    ('KFWS', 'DALLAS', 'TX', 32.573, -97.30315, 777.0),
    ('KGGW', 'GLASGOW', 'MT', 48.206361, -106.62469, 2384.0),
    ('KGJX', 'GRAND JUNCTION', 'CO', 39.062169, -108.21376, 10101.0),
    ('KGLD', 'GOODLAND', 'KS', 39.366944, -101.70027, 3716.0),
    ('KGRB', 'GREEN BAY', 'WI', 44.498633, -88.111111, 823.0),
    ('KGRK', 'FORT HOOD', 'TX', 30.721833, -97.382944, 603.0),
    ('KGRR', 'GRAND RAPIDS', 'MI', 42.893889, -85.544889, 875.0),
    ('KGSP', 'GREER', 'SC', 34.883306, -82.219833, 1069.0),
    ('KGWX', 'COLUMBUS AFB', 'MS', 33.896917, -88.329194, 590.0),
    ('KGYX', 'PORTLAND', 'ME', 43.891306, -70.256361, 474.0),
    ('KHDC', 'HAMMOND MUNICIPAL AIRPORT', 'LA', 30.5193, -90.4074, 43.0),
    ('KHDX', 'HOLLOMAN AFB', 'NM', 33.077, -106.12003, 4270.0),
    ('KHGX', 'HOUSTON', 'TX', 29.4719, -95.078733, 115.0),
    ('KHNX', 'SAN JOAQUIN VALLEY', 'CA', 36.314181, -119.63213, 340.0),
    ('KHPX', 'FORT CAMPBELL', 'KY', 36.736972, -87.285583, 613.0),
    ('KHTX', 'HUNTSVILLE', 'AL', 34.930556, -86.083611, 1859.0),
    ('KICT', 'WICHITA', 'KS', 37.654444, -97.443056, 1400.0),
    ('KICX', 'CEDAR CITY', 'UT', 37.59105, -112.86218, 10757.0),
    ('KILN', 'CINCINNATI', 'OH', 39.420483, -83.82145, 1170.0),
    ('KILX', 'LINCOLN', 'IL', 40.1505, -89.336792, 731.0),
    ('KIND', 'INDIANAPOLIS', 'IN', 39.7075, -86.280278, 887.0),
    ('KINX', 'TULSA', 'OK', 36.175131, -95.564161, 749.0),
    ('KIWA', 'PHOENIX', 'AZ', 33.289233, -111.66991, 1426.0),
    ('KIWX', 'FORT WAYNE', 'IN', 41.358611, -85.7, 1056.0),
    ('KJAX', 'JACKSONVILLE', 'FL', 30.484633, -81.7019, 160.0),
    ('KJGX', 'ROBINS AFB', 'GA', 32.675683, -83.350833, 618.0),
    ('KJKL', 'JACKSON', 'KY', 37.590833, -83.313056, 1461.0),
    ('KLBB', 'LUBBOCK', 'TX', 33.654139, -101.81416, 3378.0),
    ('KLCH', 'LAKE CHARLES', 'LA', 30.125306, -93.215889, 137.0),
    ('KLGX', 'LANGLEY HILL NW WASHINGTON', 'WA', 47.116944, -124.10666, 366.0),
    ('KLIX', 'NEW ORLEANS', 'LA', 30.336667, -89.825417, 179.0),
    ('KLNX', 'NORTH PLATTE', 'NE', 41.957944, -100.57622, 3113.0),
    ('KLOT', 'CHICAGO', 'IL', 41.604444, -88.084444, 760.0),
    ('KLRX', 'ELKO', 'NV', 40.73955, -116.8027, 6895.0),
    ('KLSX', 'ST LOUIS', 'MO', 38.698611, -90.682778, 722.0),
    ('KLTX', 'WILMINGTON', 'NC', 33.98915, -78.429108, 145.0),
    ('KLVX', 'LOUISVILLE', 'KY', 37.975278, -85.943889, 833.0),
    ('KLWX', 'STERLING', 'VA', 38.976111, -77.4875, 404.0),
    ('KLZK', 'LITTLE ROCK', 'AR', 34.8365, -92.262194, 649.0),
    ('KMAF', 'MIDLAND ODESSA', 'TX', 31.943461, -102.18925, 2962.0),
    ('KMAX', 'MEDFORD', 'OR', 42.081169, -122.71736, 7561.0),
    ('KMBX', 'MINOT AFB', 'ND', 48.393056, -100.86444, 1590.0),
    ('KMHX', 'MOREHEAD CITY', 'NC', 34.775908, -76.876189, 145.0),
    ('KMKX', 'MILWAUKEE', 'WI', 42.9679, -88.550667, 1023.0),
    ('KMLB', 'MELBOURNE', 'FL', 28.113194, -80.654083, 149.0),
    ('KMOB', 'MOBILE', 'AL', 30.679444, -88.24, 289.0),
    ('KMPX', 'MINNEAPOLIS', 'MN', 44.848889, -93.565528, 1101.0),
    ('KMQT', 'MARQUETTE', 'MI', 46.531111, -87.548333, 1525.0),
    ('KMRX', 'KNOXVILLE', 'TN', 36.168611, -83.401944, 1434.0),
    ('KMSX', 'MISSOULA', 'MT', 47.041, -113.98622, 7978.0),
    ('KMTX', 'SALT LAKE CITY', 'UT', 41.262778, -112.44777, 6594.0),
    ('KMUX', 'SAN FRANCISCO', 'CA', 37.155222, -121.89844, 3550.0),
    ('KMVX', 'GRAND FORKS', 'ND', 47.527778, -97.325556, 1083.0),
    ('KMXX', 'MAXWELL AFB', 'AL', 32.53665, -85.78975, 560.0),
    ('KNKX', 'SAN DIEGO', 'CA', 32.919017, -117.0418, 1052.0),
    ('KNQA', 'MEMPHIS', 'TN', 35.344722, -89.873333, 435.0),
    ('KOAX', 'OMAHA', 'NE', 41.320369, -96.366819, 1262.0),
    ('KOHX', 'NASHVILLE', 'TN', 36.247222, -86.5625, 676.0),
    ('KOKX', 'NEW YORK CITY', 'NY', 40.865528, -72.863917, 199.0),
    ('KOTX', 'SPOKANE', 'WA', 47.680417, -117.62677, 2449.0),
    ('KOUN', 'NORMAN NSSL', 'OK', 35.236058, -97.46235, 1284.0),
    ('KPAH', 'PADUCAH', 'KY', 37.068333, -88.771944, 506.0),
    ('KPBZ', 'PITTSBURGH', 'PA', 40.531717, -80.217967, 1266.0),
    ('KPDT', 'PENDLETON', 'OR', 45.69065, -118.85293, 1580.0),
    ('KPOE', 'FORT POLK', 'LA', 31.155278, -92.976111, 473.0),
    ('KPUX', 'PUEBLO', 'CO', 38.45955, -104.18135, 5363.0),
    ('KRAX', 'RALEIGH DURHAM', 'NC', 35.665519, -78.48975, 462.0),
    ('KRGX', 'RENO', 'NV', 39.754056, -119.46202, 8396.0),
    ('KRIW', 'RIVERTON', 'WY', 43.066089, -108.4773, 5633.0),
    ('KRLX', 'CHARLESTON', 'WV', 38.311111, -81.722778, 1213.0),
    ('KRTX', 'PORTLAND', 'OR', 45.715039, -122.965, 1728.0),
    ('KSFX', 'POCATELLO', 'ID', 43.1056, -112.68613, 4539.0),
    ('KSGF', 'SPRINGFIELD', 'MO', 37.235239, -93.400419, 1375.0),
    ('KSHV', 'SHREVEPORT', 'LA', 32.450833, -93.84125, 387.0),
    ('KSJT', 'SAN ANGELO', 'TX', 31.371278, -100.4925, 2004.0),
    ('KSOX', 'SANTA ANA MOUNTAINS', 'CA', 33.817733, -117.636, 3106.0),
    ('KSRX', 'FORT SMITH', 'AR', 35.290417, -94.361889, 737.0),
    ('KTBW', 'TAMPA', 'FL', 27.7055, -82.401778, 122.0),
    ('KTFX', 'GREAT FALLS', 'MT', 47.459583, -111.38533, 3805.0),
    ('KTLH', 'TALLAHASSEE', 'FL', 30.397583, -84.328944, 177.0),
    ('KTLX', 'OKLAHOMA CITY', 'OK', 35.333361, -97.277761, 1278.0),
    ('KTWX', 'TOPEKA', 'KS', 38.99695, -96.23255, 1415.0),
    ('KTYX', 'FORT DRUM', 'NY', 43.755694, -75.679861, 1960.0),
    ('KUDX', 'RAPID CITY', 'SD', 44.124722, -102.83, 3195.0),
    ('KUEX', 'HASTINGS', 'NE', 40.320833, -98.441944, 2057.0),
    ('KVAX', 'MOODY AFB', 'GA', 30.890278, -83.001806, 330.0),
    ('KVBX', 'VANDENBERG AFB', 'CA', 34.83855, -120.39791, 1354.0),
    ('KVNX', 'VANCE AFB', 'OK', 36.740617, -98.127717, 1258.0),
    ('KVTX', 'LOS ANGELES', 'CA', 34.412017, -119.17875, 2807.0),
    ('KVWX', 'EVANSVILLE', 'IN', 38.26025, -87.724528, 625.0),
    ('KYUX', 'YUMA', 'AZ', 32.495281, -114.65671, 239.0),
    ('LPLA', 'LAJES AB', '', 38.73028, -27.32167, 3334.0),
    ('PABC', 'BETHEL FAA', 'AK', 60.791944, -161.87638, 193.0),
    ('PACG', 'SITKA', 'AK', 56.852778, -135.52916, 272.0),
    ('PAEC', 'NOME', 'AK', 64.511389, -165.295, 90.0),
    ('PAHG', 'ANCHORAGE', 'AK', 60.725914, -151.35146, 356.0),
    ('PAIH', 'MIDDLETON ISLAND', 'AK', 59.460767, -146.30344, 132.0),
    ('PAKC', 'KING SALMON', 'AK', 58.679444, -156.62944, 144.0),
    ('PAPD', 'FAIRBANKS', 'AK', 65.035114, -147.50143, 2707.0),
    ('PGUA', 'ANDERSEN AFB AGANA', 'GU', 13.455833, 144.811111, 386.0),
    ('PHKI', 'SOUTH KAUAI', 'HI', 21.893889, -159.5525, 340.0),
    ('PHKM', 'KAMUELA', 'HI', 20.125278, -155.77777, 3966.0),
    ('PHMO', 'MOLOKAI', 'HI', 21.132778, -157.18027, 1444.0),
    ('PHWA', 'SOUTH SHORE', 'HI', 19.095, -155.56888, 1461.0),
    ('RKJK', 'KUNSAN', '', 35.924167, 126.622222, 192.0),
    ('RKSG', 'CAMP HUMPHREYS', '', 37.207569, 127.285561, 1521.0),
    ('RODN', 'KADENA', '', 26.3078, 127.903469, 412.0),
    ('TJUA', 'SAN JUAN', 'PR', 18.115667, -66.078167, 2958.0),
]

STATIONS = [RadarStation(*r) for r in _RAW_STATIONS]
STATIONS_BY_ICAO = {s.icao: s for s in STATIONS}
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    lat1r, lon1r, lat2r, lon2r = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2r - lat1r) / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin((lon2r - lon1r) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(min(1.0, a)))


def find_nearest_station_in_range(lat: float, lon: float,
                                  max_range_mi: float = 200.0) -> Tuple[Optional[RadarStation], Optional[float]]:
    """
    (nearest station, distance in miles), or (None, None) when nothing is within
    max_range_mi. Deliberately not the globally nearest site: a storm far from
    every radar gets no radar at all (MWSynth's rule, kept).
    """
    best, best_d = None, None
    for s in STATIONS:
        d = haversine_miles(lat, lon, s.lat, s.lon)
        if d <= max_range_mi and (best_d is None or d < best_d):
            best, best_d = s, d
    return best, best_d


def stations_by_distance(lat: float, lon: float, limit: int = 12) -> List[Tuple[RadarStation, float]]:
    """The nearest sites with their distances (miles), for choosing one by hand."""
    ranked = sorted(((s, haversine_miles(lat, lon, s.lat, s.lon)) for s in STATIONS), key=lambda x: x[1])
    return ranked[:limit]
