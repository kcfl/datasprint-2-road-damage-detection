"""Municipal Router & Nominatim Reverse Geocoding Module.

Translates GPS coordinates (latitude, longitude) into municipal jurisdictions
and matches them to official Indian Municipal Corporation road grievance emails.
"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import requests

MUNICIPALITIES_FILE = os.path.join(os.path.dirname(__file__), "municipalities.json")


def load_municipalities_catalog() -> Dict[str, Any]:
    """Loads seeded municipal data catalog."""
    if not os.path.exists(MUNICIPALITIES_FILE):
        return {"municipalities": [], "default": {}}
    with open(MUNICIPALITIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """Reverse geocodes latitude & longitude using OpenStreetMap Nominatim.

    Has a 4-second timeout and fails safely to structured fallback coordinates.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
    }
    headers = {
        "User-Agent": "RoadDamageCivicReporter/1.0 (Contact: civic-roads@opensource.org)"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=4.0)
        if response.status_code == 200:
            data = response.json()
            address_info = data.get("address", {})
            city_candidate = (
                address_info.get("city")
                or address_info.get("town")
                or address_info.get("municipality")
                or address_info.get("county")
                or address_info.get("state_district")
                or address_info.get("state")
                or ""
            )
            return {
                "success": True,
                "display_name": data.get("display_name", f"{lat:.5f}, {lon:.5f}"),
                "city": city_candidate,
                "suburb": address_info.get("suburb", ""),
                "road": address_info.get("road", ""),
                "state": address_info.get("state", ""),
                "postcode": address_info.get("postcode", ""),
                "country": address_info.get("country", "India"),
                "raw_address": address_info,
            }
    except Exception as e:
        print(f"[Router] Nominatim reverse-geocode failed or timed out: {e}")

    # Fallback when offline or connection blocked
    return {
        "success": False,
        "display_name": f"Location ({lat:.5f}° N, {lon:.5f}° E)",
        "city": "",
        "suburb": "",
        "road": "",
        "state": "",
        "postcode": "",
        "country": "India",
        "raw_address": {},
    }


def match_municipality(
    location_text: str = "",
    geo_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Matches text or reverse-geocoded data to seeded Indian municipal grievance contacts."""
    catalog = load_municipalities_catalog()
    municipalities = catalog.get("municipalities", [])
    default_entry = catalog.get("default", {})

    search_tokens = []

    if location_text:
        search_tokens.extend(re.split(r"[\s,.-]+", location_text.lower()))

    if geo_data:
        for field in ["city", "suburb", "road", "state", "display_name"]:
            val = str(geo_data.get(field, "")).lower()
            if val:
                search_tokens.extend(re.split(r"[\s,.-]+", val))

    clean_tokens = [t.strip() for t in search_tokens if len(t.strip()) > 2]
    combined_query_str = " ".join(clean_tokens)

    # 1. Exact or Alias matching
    for muni in municipalities:
        muni_city = muni.get("city", "").lower()
        if muni_city in combined_query_str:
            return {
                **muni,
                "matched": True,
                "match_reason": f"Matched municipal corporation for {muni['city']}",
            }

        for alias in muni.get("aliases", []):
            if alias.lower() in combined_query_str:
                return {
                    **muni,
                    "matched": True,
                    "match_reason": f"Matched locality alias '{alias}' -> {muni['name']}",
                }

    # 2. Fallback to default municipal contact
    city_display = geo_data.get("city") if geo_data and geo_data.get("city") else (location_text or "Local Ward / City")
    return {
        "id": "default_municipal_cell",
        "city": city_display,
        "name": f"{city_display} Municipal Road Works & Grievance Cell",
        "grievance_email": default_entry.get("grievance_email", "grievance-roads@municipal.gov.in"),
        "helpline": default_entry.get("helpline", "112 / Municipal Control Room"),
        "portal_url": default_entry.get("portal_url", "https://pgportal.gov.in"),
        "state": geo_data.get("state", "Local Jurisdiction") if geo_data else "Local Jurisdiction",
        "matched": False,
        "match_reason": "General municipal grievance routing (no specific seeded corporation matched)",
    }


def generate_complaint_draft(
    citizen_name: str,
    citizen_phone: str,
    citizen_email: str,
    landmark: str,
    lat: Optional[float],
    lon: Optional[float],
    address_display: str,
    detection_summary: Dict[str, Any],
    municipality_info: Dict[str, Any],
) -> Dict[str, str]:
    """Generates formal grievance complaint email subject and formatted body."""
    priority = detection_summary.get("priority", "Medium")
    damage_str = detection_summary.get("damage_summary", "Road Surface Damage")
    muni_name = municipality_info.get("name", "The Municipal Commissioner")
    city = municipality_info.get("city", "City")

    maps_url = f"https://www.google.com/maps?q={lat},{lon}" if (lat and lon) else ""
    coords_str = f"{lat:.6f}, {lon:.6f}" if (lat and lon) else "GPS coordinates not captured"

    subject = f"[{priority.upper()} PRIORITY GRIEVANCE] Urgent Road Repair Required: {damage_str} at {landmark or address_display or city}"

    counts = detection_summary.get("counts", {})
    details_list = []
    if counts.get("D40", 0) > 0:
        details_list.append(f"• Potholes (D40): {counts['D40']} (Hazard to 2-wheelers and passenger vehicles)")
    if counts.get("D20", 0) > 0:
        details_list.append(f"• Alligator Cracking (D20): {counts['D20']} (Structural sub-base failure)")
    if counts.get("D00", 0) > 0:
        details_list.append(f"• Longitudinal Cracks (D00): {counts['D00']}")
    if counts.get("D10", 0) > 0:
        details_list.append(f"• Transverse Cracks (D10): {counts['D10']}")

    breakdown_text = "\n".join(details_list) if details_list else "• Surface distress detected by automated visual audit."

    plain_body = f"""To:
The Executive Engineer (Roads & Infrastructure) / Grievance Redressal Officer
{muni_name}

Respected Sir / Madam,

SUBJECT: Citizen Grievance regarding dangerous road pavement defects at {landmark or address_display or city}.

I am lodging an official public grievance regarding defective road conditions inspected at the following location:

--------------------------------------------------
1. LOCATION DETAILS
--------------------------------------------------
• City / Municipality: {city}
• Road / Address: {address_display or 'As specified below'}
• Landmark / Locality: {landmark or 'Opposite marked location'}
• GPS Coordinates: {coords_str}
• Google Maps Location: {maps_url if maps_url else 'N/A'}

--------------------------------------------------
2. AI-INSPECTED ROAD DAMAGE ASSESSMENT
--------------------------------------------------
• Assessed Urgency Priority: {priority.upper()}
• Total Damage Zones: {detection_summary.get('total_damage_count', 0)}
• Distress Types Detected:
{breakdown_text}
• AI Risk Assessment: {detection_summary.get('urgency_note', '')}
• Inspection Model: {detection_summary.get('model_version', 'Ultralytics YOLO')}

--------------------------------------------------
3. CITIZEN COMPLAINANT DETAILS
--------------------------------------------------
• Name: {citizen_name or 'Concerned Citizen'}
• Mobile / WhatsApp: {citizen_phone or 'Not provided'}
• Email ID: {citizen_email or 'Not provided'}

I request the engineering team of {muni_name} to inspect the spot, initiate cold-mix / asphalt patch-work, and restore motorable conditions immediately to avoid fatal accidents.

An AI-annotated photographic evidence file is attached herewith for your reference and ward inspection records.

Thanking you,
Yours sincerely,

{citizen_name or 'Concerned Citizen'}
Contact: {citizen_phone or ''} | {citizen_email or ''}
(Submitted via Automated Civic Road Damage Grievance Portal)
"""

    return {
        "subject": subject,
        "body": plain_body,
        "maps_url": maps_url,
    }
