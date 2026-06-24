### **Id and Text**

| Column | Type | Description |
| --- | --- | --- |
| **id** | int | Identifier |
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
| **host_is_superhost** | bool (0/1) | If the host is premium |
| **host_listings_count** | int | Total number of properties the host has |
| **host_has_profile_pic** | bool (0/1) | if the host has a photo |
| **host_identity_verified** | bool (0/1) | if verified |
| **host_since_days** | int | Days elapsed since the host created their account (based on a fixed reference date) |
| **host_response_time_score** | int | Ordinal score (1 to 4) representing response speed (4 = within an hour, 1 = days) |
| **is_local_host** | bool (0/1) | if the host's location string contains "Berlin" |
| **has_license** | bool (0/1) | if the host has a license |

---

### **Location & Property**

| Column | Type | Description |
| --- | --- | --- |
| **neighbourhood_cleansed** | str | The specific local area |
| **neighbourhood_group_cleansed** | str | The larger neighbourhood |
| **latitude** | float | latitude |
| **longitude** | float | longitude |
| **room_type** | str | Category of the space (Entire home/apt, Private room) |
| **accommodates** | int | Maximum number of guests |
| **bedrooms** | int | Number of bedrooms |
| **beds** | int | Number of beds |
| **bathrooms_numeric** | float | Number of bathrooms |
| **amenities_count** | int | Total count of distinct amenities offered |

---

### **Availability & Reviews**

| Column | Type | Description |
| --- | --- | --- |
| **minimum_nights** | int | Minimum stay requirement |
| **maximum_nights** | int | Maximum stay allowed |
| **has_availability** | bool (0/1) | if the calendar has any bookable days |
| **availability_30** | int | Number of available days in the next 30 days |
| **availability_365** | int | Number of available days in the next 365 days |
| **number_of_reviews** | int | Total all-time reviews |
| **number_of_reviews_ltm** | int | Reviews in the Last 12 Months |
| **review_scores_rating** | float | Overall aggregate guest rating |
| **review_scores_location** | float | Specific rating for the property's location context |
| **instant_bookable** | bool (0/1) | if can book without manual host approval |

---

### **Target Variables**

| Column | Type | Description |
| --- | --- | --- |
| **price** | float | [Target] Cleaned price in standard format |
| **log_price** | float | [Target] Natural log transformation of the price (`log1p()`)  |