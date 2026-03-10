"""
backend/models/mumbai_routes.py
Single source of truth for ALL Mumbai routes and locations.
Imported by LSTM, parking, departure planner, and API.
"""

# ── 25 Mumbai Routes with coordinates ─────────────────────────────
MUMBAI_ROUTES = {
    "R001": {"name":"Western Express Hwy (Dahisar–Bandra)",  "zone":"Western Suburbs",  "lat":19.1136,"lng":72.8697, "base_speed":60, "type":"highway"},
    "R002": {"name":"SV Road (Dahisar–Bandra)",              "zone":"Western Suburbs",  "lat":19.1197,"lng":72.8464, "base_speed":35, "type":"arterial"},
    "R003": {"name":"Link Road (Malad–Andheri)",             "zone":"Western Suburbs",  "lat":19.1024,"lng":72.8384, "base_speed":30, "type":"arterial"},
    "R004": {"name":"Andheri-Kurla Road",                   "zone":"Western Suburbs",  "lat":19.1070,"lng":72.8776, "base_speed":28, "type":"arterial"},
    "R005": {"name":"JVLR (Jogeshwari–Vikhroli)",           "zone":"Central Connector","lat":19.1076,"lng":72.9044, "base_speed":45, "type":"highway"},
    "R006": {"name":"Eastern Express Hwy (Thane–Sion)",     "zone":"Eastern Suburbs",  "lat":19.0728,"lng":72.8826, "base_speed":55, "type":"highway"},
    "R007": {"name":"LBS Marg (Thane–Kurla)",               "zone":"Eastern Suburbs",  "lat":19.0896,"lng":72.9086, "base_speed":32, "type":"arterial"},
    "R008": {"name":"Ghodbunder Road (Thane)",              "zone":"Thane",            "lat":19.2183,"lng":72.9781, "base_speed":40, "type":"arterial"},
    "R009": {"name":"Mulund-Airoli Road",                   "zone":"Eastern Suburbs",  "lat":19.1672,"lng":72.9564, "base_speed":38, "type":"arterial"},
    "R010": {"name":"Sion-Panvel Highway",                  "zone":"Eastern Suburbs",  "lat":19.0412,"lng":73.0120, "base_speed":50, "type":"highway"},
    "R011": {"name":"Bandra-Worli Sea Link",                "zone":"Central Mumbai",   "lat":19.0386,"lng":72.8178, "base_speed":80, "type":"expressway"},
    "R012": {"name":"Western Railway Road (Dadar)",         "zone":"Central Mumbai",   "lat":19.0178,"lng":72.8478, "base_speed":25, "type":"local"},
    "R013": {"name":"Dr Ambedkar Road (Dadar–Parel)",       "zone":"Central Mumbai",   "lat":19.0025,"lng":72.8425, "base_speed":22, "type":"local"},
    "R014": {"name":"BKC Internal Roads",                   "zone":"BKC",              "lat":19.0631,"lng":72.8677, "base_speed":20, "type":"local"},
    "R015": {"name":"Kurla-Chembur Road",                   "zone":"Eastern Suburbs",  "lat":19.0728,"lng":72.8826, "base_speed":26, "type":"local"},
    "R016": {"name":"Eastern Freeway (Chembur–Fort)",       "zone":"South Mumbai",     "lat":18.9800,"lng":72.8344, "base_speed":70, "type":"expressway"},
    "R017": {"name":"Marine Drive",                         "zone":"South Mumbai",     "lat":18.9440,"lng":72.8232, "base_speed":45, "type":"arterial"},
    "R018": {"name":"P D Mello Road (Masjid–Sion)",        "zone":"South Mumbai",     "lat":18.9561,"lng":72.8374, "base_speed":30, "type":"arterial"},
    "R019": {"name":"Peddar Road (Kemps Corner–Worli)",     "zone":"South Mumbai",     "lat":18.9700,"lng":72.8086, "base_speed":24, "type":"local"},
    "R020": {"name":"SG Barve Marg (Churchgate–CST)",      "zone":"South Mumbai",     "lat":18.9322,"lng":72.8264, "base_speed":18, "type":"local"},
    "R021": {"name":"Palm Beach Road (Vashi–Belapur)",      "zone":"Navi Mumbai",      "lat":19.0771,"lng":73.0108, "base_speed":65, "type":"highway"},
    "R022": {"name":"Trans-Harbour Link Approach",          "zone":"Navi Mumbai",      "lat":19.0480,"lng":72.9520, "base_speed":55, "type":"highway"},
    "R023": {"name":"Thane-Belapur Road",                   "zone":"Navi Mumbai",      "lat":19.0510,"lng":73.0200, "base_speed":42, "type":"arterial"},
    "R024": {"name":"Khopoli Road (Panvel)",                "zone":"Outskirts",        "lat":18.9894,"lng":73.1175, "base_speed":58, "type":"highway"},
    "R025": {"name":"Airoli Bridge (Thane–Navi Mumbai)",    "zone":"Navi Mumbai",      "lat":19.1549,"lng":72.9987, "base_speed":50, "type":"highway"},
}

# ── Zone congestion multipliers ────────────────────────────────────
ZONE_MULT = {
    "Western Suburbs":  1.15,
    "Eastern Suburbs":  1.08,
    "Central Connector":1.05,
    "Central Mumbai":   1.20,
    "BKC":              1.25,
    "South Mumbai":     1.10,
    "Navi Mumbai":      0.80,
    "Thane":            0.90,
    "Outskirts":        0.60,
}

TYPE_MULT = {"expressway":0.50, "highway":0.75, "arterial":1.00, "local":1.30}

# ── 50 Mumbai Parking Lots (all major areas) ──────────────────────
MUMBAI_PARKING = [
    # BKC
    {"id":"P001","name":"BKC Parking Lot 1",            "lat":19.0631,"lng":72.8677,"capacity":500, "zone":"BKC"},
    {"id":"P002","name":"BKC Parking Lot 2",            "lat":19.0607,"lng":72.8701,"capacity":350, "zone":"BKC"},
    {"id":"P003","name":"MMRDA Ground Parking",          "lat":19.0654,"lng":72.8648,"capacity":800, "zone":"BKC"},
    # Western Suburbs
    {"id":"P004","name":"Andheri Station West Parking", "lat":19.1197,"lng":72.8463,"capacity":300, "zone":"Western Suburbs"},
    {"id":"P005","name":"Andheri East Multilevel",       "lat":19.1136,"lng":72.8697,"capacity":800, "zone":"Western Suburbs"},
    {"id":"P006","name":"Malad West Mindspace Parking", "lat":19.1866,"lng":72.8479,"capacity":600, "zone":"Western Suburbs"},
    {"id":"P007","name":"Goregaon Film City Parking",   "lat":19.1534,"lng":72.8618,"capacity":400, "zone":"Western Suburbs"},
    {"id":"P008","name":"Borivali Station Parking",     "lat":19.2288,"lng":72.8558,"capacity":350, "zone":"Western Suburbs"},
    {"id":"P009","name":"Dahisar Check Naka Parking",   "lat":19.2524,"lng":72.8580,"capacity":200, "zone":"Western Suburbs"},
    {"id":"P010","name":"Santacruz Airport Road Parking","lat":19.0896,"lng":72.8518,"capacity":450, "zone":"Western Suburbs"},
    # Eastern Suburbs
    {"id":"P011","name":"Ghatkopar Metro Parking",      "lat":19.0861,"lng":72.9081,"capacity":600, "zone":"Eastern Suburbs"},
    {"id":"P012","name":"Kurla Station Parking",        "lat":19.0728,"lng":72.8826,"capacity":400, "zone":"Eastern Suburbs"},
    {"id":"P013","name":"Chembur Colony Parking",       "lat":19.0522,"lng":72.8999,"capacity":300, "zone":"Eastern Suburbs"},
    {"id":"P014","name":"Mulund Check Naka Parking",    "lat":19.1688,"lng":72.9559,"capacity":250, "zone":"Eastern Suburbs"},
    {"id":"P015","name":"Vikhroli Industrial Parking",  "lat":19.1076,"lng":72.9257,"capacity":500, "zone":"Eastern Suburbs"},
    # Thane
    {"id":"P016","name":"Thane Station West Parking",   "lat":19.1890,"lng":72.9650,"capacity":700, "zone":"Thane"},
    {"id":"P017","name":"Thane Upvan Lake Parking",     "lat":19.2080,"lng":72.9620,"capacity":300, "zone":"Thane"},
    {"id":"P018","name":"Ghodbunder Road Mall Parking", "lat":19.2183,"lng":72.9781,"capacity":1000,"zone":"Thane"},
    # Central Mumbai
    {"id":"P019","name":"Dadar TT Parking",             "lat":19.0178,"lng":72.8478,"capacity":250, "zone":"Central Mumbai"},
    {"id":"P020","name":"Lower Parel Phoenix Mall",     "lat":18.9945,"lng":72.8286,"capacity":700, "zone":"Central Mumbai"},
    {"id":"P021","name":"Worli Sea Face Parking",       "lat":19.0076,"lng":72.8176,"capacity":200, "zone":"Central Mumbai"},
    {"id":"P022","name":"Bandra Station East Parking",  "lat":19.0544,"lng":72.8403,"capacity":350, "zone":"Central Mumbai"},
    {"id":"P023","name":"Bandra Linking Road Parking",  "lat":19.0607,"lng":72.8362,"capacity":400, "zone":"Central Mumbai"},
    # South Mumbai
    {"id":"P024","name":"Churchgate Station Parking",   "lat":18.9322,"lng":72.8264,"capacity":250, "zone":"South Mumbai"},
    {"id":"P025","name":"CST Station Parking",          "lat":18.9400,"lng":72.8356,"capacity":300, "zone":"South Mumbai"},
    {"id":"P026","name":"Nariman Point Multilevel",     "lat":18.9255,"lng":72.8242,"capacity":600, "zone":"South Mumbai"},
    {"id":"P027","name":"Colaba Causeway Parking",      "lat":18.9067,"lng":72.8147,"capacity":200, "zone":"South Mumbai"},
    {"id":"P028","name":"Fort Market Parking",          "lat":18.9338,"lng":72.8356,"capacity":180, "zone":"South Mumbai"},
    {"id":"P029","name":"Cuffe Parade Parking",         "lat":18.9040,"lng":72.8182,"capacity":350, "zone":"South Mumbai"},
    {"id":"P030","name":"Marine Lines Parking",         "lat":18.9440,"lng":72.8232,"capacity":150, "zone":"South Mumbai"},
    # Navi Mumbai
    {"id":"P031","name":"Vashi Station Parking",        "lat":19.0771,"lng":73.0108,"capacity":500, "zone":"Navi Mumbai"},
    {"id":"P032","name":"Belapur CBD Parking",          "lat":19.0190,"lng":73.0381,"capacity":600, "zone":"Navi Mumbai"},
    {"id":"P033","name":"Nerul Station Parking",        "lat":19.0369,"lng":73.0169,"capacity":400, "zone":"Navi Mumbai"},
    {"id":"P034","name":"Kharghar Hills Parking",       "lat":19.0474,"lng":73.0659,"capacity":300, "zone":"Navi Mumbai"},
    {"id":"P035","name":"Airoli Sector Parking",        "lat":19.1549,"lng":72.9987,"capacity":450, "zone":"Navi Mumbai"},
    # Powai / Vikhroli
    {"id":"P036","name":"Powai Hiranandani Parking",   "lat":19.1136,"lng":72.9060,"capacity":800, "zone":"Eastern Suburbs"},
    {"id":"P037","name":"Powai Lake Parking",           "lat":19.1234,"lng":72.9071,"capacity":200, "zone":"Eastern Suburbs"},
    # Hospitals / Special
    {"id":"P038","name":"Lilavati Hospital Parking",    "lat":19.0486,"lng":72.8282,"capacity":200, "zone":"Western Suburbs"},
    {"id":"P039","name":"KEM Hospital Parking",         "lat":18.9946,"lng":72.8430,"capacity":150, "zone":"Central Mumbai"},
    {"id":"P040","name":"Wankhede Stadium Parking",     "lat":18.9383,"lng":72.8247,"capacity":300, "zone":"South Mumbai"},
    # Airports
    {"id":"P041","name":"Mumbai Airport T1 Parking",   "lat":19.0896,"lng":72.8518,"capacity":1500,"zone":"Western Suburbs"},
    {"id":"P042","name":"Mumbai Airport T2 Parking",   "lat":19.0993,"lng":72.8664,"capacity":2000,"zone":"Western Suburbs"},
    # Malls
    {"id":"P043","name":"Infinity Mall Andheri",        "lat":19.1375,"lng":72.8329,"capacity":600, "zone":"Western Suburbs"},
    {"id":"P044","name":"R City Mall Ghatkopar",        "lat":19.0861,"lng":72.9086,"capacity":700, "zone":"Eastern Suburbs"},
    {"id":"P045","name":"Viviana Mall Thane",            "lat":19.2183,"lng":72.9613,"capacity":900, "zone":"Thane"},
    {"id":"P046","name":"Seawoods Grand Central Mall",  "lat":19.0145,"lng":73.0199,"capacity":800, "zone":"Navi Mumbai"},
    {"id":"P047","name":"Palladium Mall Lower Parel",   "lat":18.9930,"lng":72.8268,"capacity":600, "zone":"Central Mumbai"},
    {"id":"P048","name":"Oberoi Mall Goregaon",         "lat":19.1534,"lng":72.8511,"capacity":700, "zone":"Western Suburbs"},
    {"id":"P049","name":"High Street Phoenix Kurla",    "lat":19.0728,"lng":72.8761,"capacity":500, "zone":"Eastern Suburbs"},
    {"id":"P050","name":"Inorbit Mall Vashi",            "lat":19.0785,"lng":73.0139,"capacity":650, "zone":"Navi Mumbai"},
]

# ── Helper: find nearest route to coordinates ──────────────────────
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * asin(sqrt(a))

def nearest_routes(lat, lng, top_k=3):
    """Return top_k nearest routes to given coordinates."""
    dists = []
    for rid, info in MUMBAI_ROUTES.items():
        d = haversine(lat, lng, info["lat"], info["lng"])
        dists.append((d, rid, info))
    dists.sort()
    return [(rid, info) for _, rid, info in dists[:top_k]]

def nearest_parking(lat, lng, top_k=5, max_km=2.0):
    """Return parking lots within max_km of given coordinates."""
    results = []
    for lot in MUMBAI_PARKING:
        d = haversine(lat, lng, lot["lat"], lot["lng"])
        if d <= max_km:
            results.append((d, lot))
    results.sort(key=lambda x: x[0])
    return [(round(d,2), lot) for d, lot in results[:top_k]]