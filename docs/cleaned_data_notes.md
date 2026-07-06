### **Id and Text**

| Column | Type | Description |
| --- | --- | --- |
| **listing_id** | int | Identifier |
| **host_id** | int | Identifier for the host |
| **name** | str | Title of the listing |
| **description** | str | Property description. Missing values filled with "" |
| **neighborhood_overview** | str | Host's description of the area |
| **host_about** | str | Host's bio |

---

### **Host Metrics**

| Column | Type | Description |
| --- | --- | --- |
| **host_response_rate** | float | Percentage of messages the host responds to |
| **host_acceptance_rate** | float | Percentage of booking requests the host accepts |
| **is_superhost** | bool (0/1) | If the host is premium |
| **host_listings_count** | int | Total number of properties the host has |
| **host_has_pic** | bool (0/1) | if the host has a photo |
| **is_host_verified** | bool (0/1) | if verified |
| **host_since_days** | int | Days elapsed since the host created their account (based on a fixed reference date) |
| **response_time_score** | int | Ordinal score (1 to 4) representing response speed (4 = within an hour, 1 = days) |
| **is_local_host** | bool (0/1) | if the host's location string contains "Berlin" |
| **has_license** | bool (0/1) | if the host has a license |

---

### **Location and Property**

| Column | Type | Description |
| --- | --- | --- |
| **neighborhood** | str | The specific local area |
| **neighborhood_group** | str | The larger neighbourhood |
| **latitude** | float | latitude |
| **longitude** | float | longitude |
| **accommodates** | int | Maximum number of guests |
| **bedrooms** | int | Number of bedrooms |
| **beds** | int | Number of beds |
| **bathrooms** | float | Number of bathrooms |
| **amenities_count** | int | Total count of distinct amenities offered |

---

### **Room Type**

| Column | Type | Description |
| --- | --- | --- |
| **room_type_entire_home** | bool (0/1) | entire home/apartment |
| **room_type_hotel** | bool (0/1) | hotel room |
| **room_type_private** | bool (0/1) | private room |
| **room_type_shared** | bool (0/1) | shared room |

---

### **Availability and Reviews**

| Column | Type | Description |
| --- | --- | --- |
| **min_nights** | int | Minimum stay requirement |
| **max_nights** | int | Maximum stay allowed |
| **has_availability** | bool (0/1) | if the calendar has any bookable days |
| **avail_30** | int | Number of available days in the next 30 days |
| **avail_365** | int | Number of available days in the next 365 days |
| **total_reviews** | int | Total all-time reviews |
| **reviews_last_12m** | int | Reviews in the Last 12 Months |
| **rating_overall** | float | Overall aggregate guest rating |
| **rating_overall** | bool (0/1) | 1 if rating_overall was empty (was replaced by train mean in data_loader) |
| **rating_location** | float | Specific rating for the property's location context |
| **instant_bookable** | bool (0/1) | if can book without manual host approval |

---

### **Target Variables**

| Column | Type | Description |
| --- | --- | --- |
| **price** | float | [Target] Cleaned price in standard format |
| **log_price** | float | [Target] Natural log transformation of the price (`log1p()`)  |