import os
import threading
import time
import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import requests
import pandas as pd
import uvicorn
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import logging
from math import sqrt, radians, cos, sin, asin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. FastAPI app
# -------------------------------
app = FastAPI(title="Places Emotion Recommender API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2. MongoDB setup - Use environment variable for production
# -------------------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb+srv://navyasree:Jungkook1!@cloudwatch.tom4vt5.mongodb.net/")
client = MongoClient(MONGO_URI)
db = client["places_db"]
places_collection = db["nearby_places"]
# New collection for location cache
location_cache_collection = db["location_cache"]

# -------------------------------
# 3. Location caching utility functions (NEW)
# -------------------------------
def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def get_cached_location():
    """
    Get the last cached location from database
    """
    try:
        cached = location_cache_collection.find_one({}, sort=[("timestamp", -1)])
        if cached:
            return {
                "latitude": cached.get("latitude"),
                "longitude": cached.get("longitude"),
                "timestamp": cached.get("timestamp"),
                "places_count": cached.get("places_count", 0)
            }
        return None
    except Exception as e:
        print(f"Error getting cached location: {e}")
        return None

def save_location_cache(latitude: float, longitude: float, places_count: int):
    """
    Save current location to cache
    """
    try:
        cache_data = {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": pd.Timestamp.now().isoformat(),
            "places_count": places_count
        }
        location_cache_collection.insert_one(cache_data)
        print(f"Saved location cache: {latitude}, {longitude} with {places_count} places")
    except Exception as e:
        print(f"Error saving location cache: {e}")

def is_within_proximity(current_lat: float, current_lon: float, cached_lat: float, cached_lon: float, proximity_km: float = 1.0) -> bool:
    """
    Check if current location is within proximity of cached location
    """
    distance = calculate_distance_km(current_lat, current_lon, cached_lat, cached_lon)
    print(f"Distance from cached location: {distance:.2f} km")
    return distance <= proximity_km

# -------------------------------
# 4. Enhanced Place Information Fetcher
# -------------------------------
def get_place_details_from_overpass(place_id: str, place_type: str) -> Dict[str, str]:
    """
    Fetch detailed information about a place using Overpass API
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Query for detailed information
    query = f"""
[out:json][timeout:15];
{place_type}({place_id});
out tags;
"""
    
    try:
        response = requests.get(overpass_url, params={"data": query}, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if data.get("elements"):
            tags = data["elements"][0].get("tags", {})
            
            # Extract description from various tag sources
            description_parts = []
            
            # Primary description sources
            if tags.get("description"):
                description_parts.append(tags["description"])
            if tags.get("tourism:description"):
                description_parts.append(tags["tourism:description"])
            if tags.get("note"):
                description_parts.append(tags["note"])
                
            # Build contextual description from tags
            context_parts = []
            
            # Cuisine information
            if tags.get("cuisine"):
                context_parts.append(f"serves {tags['cuisine']} cuisine")
            
            # Service/facility information
            if tags.get("amenity"):
                amenity = tags["amenity"]
                if amenity == "restaurant":
                    context_parts.append("dining establishment")
                elif amenity == "cafe":
                    context_parts.append("coffee shop and casual dining")
                elif amenity == "hospital":
                    context_parts.append("medical care facility")
                elif amenity == "bank":
                    context_parts.append("financial services")
                elif amenity == "pharmacy":
                    context_parts.append("medication and health products")
                elif amenity == "school":
                    context_parts.append("educational institution")
                elif amenity == "library":
                    context_parts.append("books and study facility")
                elif amenity == "gym":
                    context_parts.append("fitness and exercise facility")
                else:
                    context_parts.append(f"{amenity} facility")
            
            # Tourism information
            if tags.get("tourism"):
                tourism = tags["tourism"]
                if tourism == "hotel":
                    context_parts.append("accommodation and lodging")
                elif tourism == "museum":
                    context_parts.append("cultural exhibitions and artifacts")
                elif tourism == "attraction":
                    context_parts.append("tourist destination and sightseeing")
                elif tourism == "viewpoint":
                    context_parts.append("scenic overlook with views")
                else:
                    context_parts.append(f"{tourism} destination")
            
            # Leisure information
            if tags.get("leisure"):
                leisure = tags["leisure"]
                if leisure == "park":
                    context_parts.append("outdoor recreation and nature")
                elif leisure == "sports_centre":
                    context_parts.append("sports activities and fitness")
                elif leisure == "swimming_pool":
                    context_parts.append("swimming and water activities")
                elif leisure == "garden":
                    context_parts.append("landscaped outdoor space")
                else:
                    context_parts.append(f"{leisure} activity")
            
            # Shop information
            if tags.get("shop"):
                shop = tags["shop"]
                if shop == "mall":
                    context_parts.append("shopping center with multiple stores")
                elif shop == "supermarket":
                    context_parts.append("grocery and daily necessities")
                elif shop == "clothes":
                    context_parts.append("clothing and fashion retail")
                elif shop == "book":
                    context_parts.append("books and reading materials")
                else:
                    context_parts.append(f"{shop} retail store")
            
            # Additional contextual information
            if tags.get("building"):
                building_type = tags["building"]
                if building_type in ["church", "temple", "mosque", "synagogue"]:
                    context_parts.append("place of worship and spiritual activities")
                elif building_type == "hospital":
                    context_parts.append("medical treatment and healthcare")
                elif building_type == "school":
                    context_parts.append("learning and educational programs")
            
            # Combine all description parts
            full_description = " ".join(description_parts)
            if context_parts:
                contextual_info = ", ".join(context_parts)
                if full_description:
                    full_description += f". {contextual_info}"
                else:
                    full_description = contextual_info
            
            return {
                "description": full_description,
                "raw_tags": tags
            }
    except Exception as e:
        print(f"Could not fetch detailed info: {e}")
        return {"description": "", "raw_tags": {}}
    
    return {"description": "", "raw_tags": {}}

# -------------------------------
# 5. Enhanced Emotion Analysis for Places
# -------------------------------
class PlaceEmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.reference_data = None
        self._load_model()
        self._load_reference_data()
    
    def _load_model(self):
        try:
            # Using a more context-aware model for better understanding
            self.model = SentenceTransformer("all-mpnet-base-v2")
            print("Loaded context-aware emotion analysis model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_reference_data(self):
        # Try to load from data directory first, then fallback to current directory
        possible_paths = [
            "data/df_result.xlsx",  # Render deployment path
            "df_result.xlsx",       # Fallback
            os.path.join(os.getcwd(), "data", "df_result.xlsx")  # Local development
        ]
        
        for file_path in possible_paths:
            try:
                if os.path.exists(file_path):
                    df_result = pd.read_excel(file_path)
                    df_result = df_result.dropna(subset=["category_combined", "place_emotion"])
                    
                    if "place_description" in df_result.columns:
                        df_result['text_for_embedding'] = (
                            df_result['category_combined'].astype(str) + " " + 
                            df_result['place_description'].astype(str)
                        )
                    else:
                        df_result['text_for_embedding'] = df_result['category_combined'].astype(str)
                    
                    print("Computing embeddings for reference data...")
                    self.reference_embeddings = self.model.encode(
                        df_result['text_for_embedding'].tolist(), 
                        convert_to_tensor=True
                    )
                    self.reference_data = df_result
                    print(f"Loaded {len(df_result)} reference places with emotions from {file_path}")
                    return
            except Exception as e:
                print(f"Error loading reference data from {file_path}: {e}")
                continue
        
        print("Reference dataset not found, using enhanced emotion mapping")
        self.reference_data = None
    
    def create_context_aware_text(self, place_name: str, place_category: str, place_description: str = None, tags: Dict = None) -> str:
        """
        Create a rich, context-aware text representation of the place
        """
        # Start with name and category
        context_text = f"{place_name} is a {place_category}"
        
        # Add description if available
        if place_description and place_description.strip():
            context_text += f". {place_description}"
        
        # Add contextual information from tags
        if tags:
            context_additions = []
            
            # Add opening hours context
            if tags.get("opening_hours"):
                context_additions.append("operates with specific hours")
            
            # Add accessibility info
            if tags.get("wheelchair") == "yes":
                context_additions.append("wheelchair accessible")
            
            # Add atmosphere indicators
            if tags.get("outdoor_seating") == "yes":
                context_additions.append("offers outdoor seating")
            if tags.get("takeaway") == "yes":
                context_additions.append("provides takeaway service")
            if tags.get("wifi") == "yes":
                context_additions.append("has wifi connectivity")
            
            # Add price indicators
            if tags.get("price_range"):
                price = tags["price_range"]
                if price in ["$", "cheap"]:
                    context_additions.append("budget-friendly pricing")
                elif price in ["$$", "moderate"]:
                    context_additions.append("moderate pricing")
                elif price in ["$$$", "$$$$", "expensive"]:
                    context_additions.append("upscale pricing")
            
            # Add brand/chain context
            if tags.get("brand"):
                context_additions.append(f"part of {tags['brand']} chain")
            
            if context_additions:
                context_text += f". Features include: {', '.join(context_additions)}"
        
        return context_text
    
    def predict_emotions_for_place(self, place_name: str, place_category: str, 
                                   place_description: str = None, tags: Dict = None) -> List[Dict]:
        """
        Enhanced emotion prediction using context-aware text analysis
        """
        # Create rich context-aware text
        context_aware_text = self.create_context_aware_text(
            place_name, place_category, place_description, tags
        )
        
        print(f"Context text for {place_name}: {context_aware_text[:100]}...")
        
        try:
            if self.reference_data is not None:
                return self._get_emotions_from_similarity(context_aware_text)
            else:
                return self._get_enhanced_emotions_from_context(context_aware_text, place_category, tags)
        except Exception as e:
            print(f"Error predicting emotions for {place_name}: {e}")
            return [{"emotion": "neutral", "confidence": 0.5}]
    
    def _get_emotions_from_similarity(self, context_text: str, top_k: int = 3) -> List[Dict]:
        """
        Use semantic similarity with reference data for emotion prediction
        """
        place_embedding = self.model.encode(context_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(place_embedding, self.reference_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=min(top_k * 3, len(cosine_scores)))
        
        emotion_scores = {}
        for score, idx in zip(top_results.values, top_results.indices):
            if score.item() > 0.2:  # Lower threshold for more nuanced matching
                emotion = self.reference_data.iloc[idx.item()]['place_emotion']
                confidence = float(score.item())
                if emotion in emotion_scores:
                    emotion_scores[emotion] = max(emotion_scores[emotion], confidence)
                else:
                    emotion_scores[emotion] = confidence
        
        emotions = []
        for emotion, confidence in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            emotions.append({"emotion": emotion, "confidence": round(confidence, 3)})
        
        return emotions if emotions else [{"emotion": "neutral", "confidence": 0.5}]
    
    def _get_enhanced_emotions_from_context(self, context_text: str, category: str, tags: Dict = None) -> List[Dict]:
        emotions = []
        context_lower = context_text.lower()
        
        if any(word in context_lower for word in ["temple", "church", "mosque", "worship", "spiritual", "shrine", "monastery", "cathedral"]):
            emotions.append({"emotion": "spirituality", "confidence": 0.9})
            emotions.append({"emotion": "contemplative", "confidence": 0.8})
            emotions.append({"emotion": "peaceful", "confidence": 0.7})
        
        if any(word in context_lower for word in ["park", "garden", "outdoor", "nature", "scenic", "viewpoint", "spa", "wellness"]):
            emotions.append({"emotion": "calm", "confidence": 0.9})
            emotions.append({"emotion": "peaceful", "confidence": 0.8})
            emotions.append({"emotion": "relaxed", "confidence": 0.8})
        
        if any(word in context_lower for word in ["restaurant", "cafe", "dining", "food", "cuisine"]):
            emotions.append({"emotion": "social", "confidence": 0.8})
            emotions.append({"emotion": "comfort", "confidence": 0.7})
            if "upscale" in context_lower or "fine dining" in context_lower:
                emotions.append({"emotion": "luxury", "confidence": 0.8})
        
        if any(word in context_lower for word in ["museum", "cultural", "exhibition", "library", "educational"]):
            emotions.append({"emotion": "curious", "confidence": 0.8})
            emotions.append({"emotion": "contemplative", "confidence": 0.7})
        
        if any(word in context_lower for word in ["gym", "fitness", "sports", "exercise", "swimming"]):
            emotions.append({"emotion": "energetic", "confidence": 0.9})
            emotions.append({"emotion": "confident", "confidence": 0.7})
        
        if any(word in context_lower for word in ["shop", "mall", "retail", "store"]):
            emotions.append({"emotion": "excitement", "confidence": 0.7})
            emotions.append({"emotion": "social", "confidence": 0.6})
        
        if any(word in context_lower for word in ["cinema", "theater", "entertainment", "attraction"]):
            emotions.append({"emotion": "excitement", "confidence": 0.8})
            emotions.append({"emotion": "joy", "confidence": 0.7})
        
        if any(word in context_lower for word in ["hotel", "accommodation", "lodging"]):
            emotions.append({"emotion": "comfort", "confidence": 0.8})
            emotions.append({"emotion": "relaxed", "confidence": 0.7})
        
        if not emotions:
            emotions = [{"emotion": "neutral", "confidence": 0.5}]
        
        return sorted(emotions, key=lambda x: x["confidence"], reverse=True)[:3]

emotion_analyzer = PlaceEmotionAnalyzer()

# -------------------------------
# 6. Input models
# -------------------------------
class Location(BaseModel):
    latitude: float
    longitude: float

class UserPreference(BaseModel):
    latitude: float
    longitude: float
    emotions: List[str]
    vector: List[int] = None
    timestamp: str = None

# -------------------------------
# 7. Enhanced fetch nearby places with detailed context and caching (MODIFIED)
# -------------------------------
def get_nearby_places_with_emotions(lat, lon, radius=1000):  # Changed to 1km radius
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Enhanced query to get more detailed information
    query = f"""
[out:json][timeout:30];
(
  node(around:{radius},{lat},{lon})[amenity][name];
  node(around:{radius},{lat},{lon})[tourism][name];
  node(around:{radius},{lat},{lon})[leisure][name];
  node(around:{radius},{lat},{lon})[shop][name];
  way(around:{radius},{lat},{lon})[amenity][name];
  way(around:{radius},{lat},{lon})[tourism][name];
  way(around:{radius},{lat},{lon})[leisure][name];
  way(around:{radius},{lat},{lon})[shop][name];
);
out center tags 100;
"""

    try:
        print("Fetching places from OpenStreetMap with detailed information...")
        response = requests.get(overpass_url, params={"data": query}, timeout=40)
        response.raise_for_status()
        data = response.json()
        print(f"Found {len(data.get('elements', []))} raw places")
    except Exception as e:
        print(f"Error fetching places: {e}")
        return []
    
    places_with_emotions = []
    processed_names = set()
    
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name", "").strip()
        
        # Skip places without names or already processed
        if not name or name in processed_names:
            continue
        processed_names.add(name)
        
        # Get primary category
        category = (tags.get("amenity") or tags.get("tourism") or 
                    tags.get("leisure") or tags.get("shop") or "unknown")
        
        # Skip unknown categories
        if category.lower() == "unknown":
            continue
        
        # Get coordinates
        lat_val = element.get("lat") or element.get("center", {}).get("lat")
        lon_val = element.get("lon") or element.get("center", {}).get("lon")
        
        # Create initial description from tags
        initial_description = f"{name} - {category}"
        if tags.get("cuisine"):
            initial_description += f" ({tags.get('cuisine')})"
        
        # Get enhanced place details
        element_type = element.get("type", "node")
        element_id = element.get("id")
        
        print(f"Fetching detailed info for: {name}")
        place_details = get_place_details_from_overpass(element_id, element_type)
        
        # Combine name with enhanced description
        enhanced_description = place_details.get("description", "")
        if not enhanced_description:
            enhanced_description = initial_description
        else:
            enhanced_description = f"{initial_description}. {enhanced_description}"
        
        # Create combined context text for emotion analysis
        print(f"Analyzing emotions for: {name}")
        emotions = emotion_analyzer.predict_emotions_for_place(
            name, category, enhanced_description, tags
        )
        
        emotion_vector = {e["emotion"]: e["confidence"] for e in emotions}
        
        place_data = {
            "name": name,
            "category": category,
            "description": enhanced_description,
            "combined_context": f"{name} {enhanced_description}",
            "lat": lat_val,
            "lon": lon_val,
            "emotions": emotions,
            "emotion_vector": emotion_vector,
            "osm_tags": tags,
            "raw_details": place_details.get("raw_tags", {}),
            "created_at": pd.Timestamp.now().isoformat()
        }
        places_with_emotions.append(place_data)
    
    print(f"Processed {len(places_with_emotions)} unique places with context-aware emotions")
    return places_with_emotions

# -------------------------------
# 8. API endpoints with location caching (MODIFIED)
# -------------------------------
@app.post("/fetch_places")
def fetch_and_store_places_with_emotions(loc: Location):
    try:
        lat, lon = loc.latitude, loc.longitude
        print(f"Fetch request for coordinates: {lat}, {lon}")
        
        # Check cached location
        cached_location = get_cached_location()
        
        if cached_location:
            cached_lat = cached_location["latitude"]
            cached_lon = cached_location["longitude"]
            places_count = cached_location["places_count"]
            
            print(f"Found cached location: {cached_lat}, {cached_lon} with {places_count} places")
            
            # Check if current location is within 1km of cached location
            if is_within_proximity(lat, lon, cached_lat, cached_lon, 1.0):
                print("Current location is within 1km of cached location - using existing data")
                existing_places = list(places_collection.find({}, {"_id": 0}))
                
                if existing_places:
                    return {
                        "status": "success",
                        "message": "Using cached places data (within 1km proximity)",
                        "total_places": len(existing_places),
                        "cache_used": True,
                        "distance_from_cache": round(calculate_distance_km(lat, lon, cached_lat, cached_lon), 2)
                    }
        
        # If we reach here, either no cache exists or user moved >1km away
        print("Fetching new places data...")
        places = get_nearby_places_with_emotions(lat, lon)
        
        if not places:
            print("No places found nearby")
            return {"status": "success", "message": "No places found nearby", "total_places": 0, "cache_used": False}
        
        # Clear existing data and store new places
        places_collection.delete_many({})
        result = places_collection.insert_many(places)
        
        # Save new location to cache
        save_location_cache(lat, lon, len(places))
        
        print(f"Stored {len(result.inserted_ids)} new places in database")
        
        # Save to Excel if possible (optional for cloud deployment)
        try:
            df = pd.DataFrame(places)
            excel_path = os.path.join(os.getcwd(), "places_with_enhanced_emotions.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"Saved enhanced report to: {excel_path}")
        except Exception as e:
            print(f"Could not save Excel file: {e}")
            excel_path = "N/A"
        
        return {
            "status": "success",
            "message": "Places fetched with context-aware emotion analysis",
            "total_places": len(places),
            "cache_used": False,
            "excel_file": excel_path
        }
    except Exception as e:
        print(f"Error in fetch_and_store_places_with_emotions: {e}")
        return {"status": "error", "message": str(e), "total_places": 0, "cache_used": False}

@app.get("/places")
def get_all_places_with_emotions():
    try:
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "success", "message": "No places found", "places": [], "count": 0}
        
        all_emotions = set()
        emotion_counts = {}
        for place in places:
            for e in place.get("emotions", []):
                all_emotions.add(e["emotion"])
                emotion_counts[e["emotion"]] = emotion_counts.get(e["emotion"], 0) + 1
        
        return {
            "status": "success",
            "places": places,
            "count": len(places),
            "emotion_summary": {"unique_emotions": list(all_emotions), "emotion_frequency": emotion_counts}
        }
    except Exception as e:
        print(f"Error retrieving places: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_emotion/{place_name}")
def test_single_emotion_analysis(place_name: str, category: str = "restaurant", description: str = ""):
    try:
        emotions = emotion_analyzer.predict_emotions_for_place(place_name, category, description)
        emotion_vector = {e["emotion"]: e["confidence"] for e in emotions}
        context_text = emotion_analyzer.create_context_aware_text(place_name, category, description)
        return {
            "place_name": place_name, 
            "category": category, 
            "description": description,
            "context_text": context_text,
            "predicted_emotions": emotions, 
            "emotion_vector": emotion_vector
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug_place/{place_name}")
def debug_place_matching(place_name: str, user_emotions: str = "Spirituality,Relaxation/Calm"):
    """
    Debug a specific place's emotion matching
    """
    try:
        # Parse user emotions
        emotions_list = [e.strip() for e in user_emotions.split(',')]
        
        # Find the place in database
        place = places_collection.find_one({"name": {"$regex": place_name, "$options": "i"}}, {"_id": 0})
        
        if not place:
            return {"error": f"Place '{place_name}' not found"}
        
        # Calculate match score
        match_info = calculate_emotion_match_score(emotions_list, place.get("emotion_vector", {}))
        
        return {
            "place_name": place.get("name"),
            "place_emotions": place.get("emotions", []),
            "emotion_vector": place.get("emotion_vector", {}),
            "user_emotions": emotions_list,
            "match_info": match_info,
            "would_pass_filtering": match_info["final_score"] > 0.01,
            "description": place.get("description", "")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/cache_status")
def get_cache_status():
    """
    Get current cache status and location information (NEW)
    """
    try:
        cached_location = get_cached_location()
        places_count = places_collection.count_documents({})
        
        if cached_location:
            return {
                "status": "success",
                "has_cache": True,
                "cached_location": {
                    "latitude": cached_location["latitude"],
                    "longitude": cached_location["longitude"],
                    "timestamp": cached_location["timestamp"],
                    "places_count": cached_location["places_count"]
                },
                "current_places_in_db": places_count,
                "cache_valid": places_count > 0
            }
        else:
            return {
                "status": "success",
                "has_cache": False,
                "cached_location": None,
                "current_places_in_db": places_count,
                "cache_valid": False
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/clear_cache")
def clear_location_cache():
    """
    Clear the location cache (for testing purposes) (NEW)
    """
    try:
        location_cache_collection.delete_many({})
        places_collection.delete_many({})
        return {
            "status": "success",
            "message": "Location cache and places data cleared"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health_check():
    places_count = places_collection.count_documents({})
    cached_location = get_cached_location()  # MODIFIED
    
    return {
        "status": "healthy",
        "emotion_model_loaded": emotion_analyzer.model is not None,
        "reference_data_loaded": emotion_analyzer.reference_data is not None,
        "places_in_database": places_count,
        "has_location_cache": cached_location is not None,  # NEW
        "cache_location": cached_location,  # NEW
        "ready_for_recommendations": places_count > 0,
        "model_type": "Context-aware SentenceTransformer with Location Caching"  # MODIFIED
    }

# -------------------------------
# 9. Improved Recommendation System
# -------------------------------
def calculate_emotion_match_score(user_emotions: List[str], place_emotion_vector: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate detailed matching scores between user emotions and place emotions
    """
    # Map frontend emotion names to backend emotion names
    emotion_mapping = {
    "Joy/Happy": ["joy, happy", "joy", "happy", "entertainment"],
    "Relaxation/Calm": ["relaxation,calm", "relaxation", "calm", "comfort", "wellness"],
    "Social Emotion": ["social", "food", "entertainment"],
    "Excitement": ["excitement", "adventure", "entertainment"],
    "Comfort": ["comfort", "wellness"],
    "Adventure": ["adventure", "excitement", "energy"],
    "Romance": ["luxury", "comfort"],  # No direct romance emotion found
    "Luxury": ["luxury", "comfort"],
    "Shopping": ["shopping", "retail", "convenience"],
    "Nostalgia": ["comfort"],  # No direct nostalgia emotion found
    "Energy": ["energy", "excitement", "adventure","wellness"],
    "Wellness": ["wellness", "healthcare"],
    "Entertainment": ["entertainment", "excitement", "joy, happy"],
    "Exploration": ["adventure", "education", "excitement"],
    "Fear": ["fear", "stress"],
    "Creativity": ["education", "entertainment"],  # No direct creativity emotion found
    "Spirituality": ["spirituality","relaxation,calm" ],
    "Education": ["education", "professional"],
    "Retail": ["retail", "shopping", "convenience"],
    "Outdoors": ["adventure", "energy", "excitement"]
}
    
    # Flatten user emotions to backend emotion names
    user_backend_emotions = []
    for emotion in user_emotions:
        mapped = emotion_mapping.get(emotion, [emotion.lower()])
        user_backend_emotions.extend(mapped)
    
    # Remove neutral emotions from place vector for scoring
    filtered_place_vector = {k: v for k, v in place_emotion_vector.items() 
                           if k != "neutral" and v > 0.2}
    
    if not filtered_place_vector:
        return {
            "exact_matches": 0,
            "exact_match_score": 0.0,
            "weighted_score": 0.0,
            "coverage_score": 0.0,
            "final_score": 0.0,
            "matched_emotions": []
        }
    
    # Calculate exact matches
    exact_matches = []
    exact_match_score = 0.0
    
    for user_emotion in user_backend_emotions:
        if user_emotion in filtered_place_vector:
            confidence = filtered_place_vector[user_emotion]
            exact_matches.append((user_emotion, confidence))
            exact_match_score += confidence
    
    # Calculate weighted score (prefers places with multiple matching emotions)
    weighted_score = exact_match_score * (1 + 0.5 * len(exact_matches))
    
    # Calculate coverage score (how many of user's emotions are covered)
    coverage_score = len(exact_matches) / len(set(user_backend_emotions)) if user_backend_emotions else 0
    
    # Calculate final score combining all factors
    final_score = (
        exact_match_score * 0.4 +
        weighted_score * 0.4 +
        coverage_score * 0.2
    )
    
    return {
        "exact_matches": len(exact_matches),
        "exact_match_score": round(exact_match_score, 3),
        "weighted_score": round(weighted_score, 3),
        "coverage_score": round(coverage_score, 3),
        "final_score": round(final_score, 3),
        "matched_emotions": exact_matches
    }

def rank_places_by_emotion_priority(places: List[Dict], user_emotions: List[str]) -> List[Dict]:
    """
    Rank places with priority system - only include places where user emotions are dominant
    """
    scored_places = []
    
    print(f"Starting with {len(places)} places")
    print(f"User selected emotions: {user_emotions}")
    
    # Map frontend emotion names to backend emotion names
    emotion_mapping = {
    "Joy/Happy": ["joy, happy", "joy", "happy", "entertainment"],
    "Relaxation/Calm": ["relaxation,calm", "relaxation", "calm", "comfort", "wellness"],
    "Social Emotion": ["social", "food", "entertainment"],
    "Excitement": ["excitement", "adventure", "entertainment"],
    "Comfort": ["comfort", "wellness"],
    "Adventure": ["adventure", "excitement", "energy"],
    "Romance": ["luxury", "comfort"],  # No direct romance emotion found
    "Luxury": ["luxury", "comfort"],
    "Shopping": ["shopping", "retail", "convenience"],
    "Nostalgia": ["comfort"],  # No direct nostalgia emotion found
    "Energy": ["energy", "excitement", "adventure","wellness"],
    "Wellness": ["wellness", "healthcare"],
    "Entertainment": ["entertainment", "excitement", "joy, happy"],
    "Exploration": ["adventure", "education", "excitement"],
    "Fear": ["fear", "stress"],
    "Creativity": ["education", "entertainment"],  # No direct creativity emotion found
    "Spirituality": ["spirituality","relaxation,calm"],
    "Education": ["education", "professional"],
    "Retail": ["retail", "shopping", "convenience"],
    "Outdoors": ["adventure", "energy", "excitement"]
}
    
    # Flatten user emotions to backend emotion names
    user_backend_emotions = set()
    for emotion in user_emotions:
        mapped = emotion_mapping.get(emotion, [emotion.lower()])
        user_backend_emotions.update(mapped)
    
    for place in places:
        emotion_vector = place.get("emotion_vector", {})
        place_name = place.get('name', 'Unknown')
        
        print(f"\nAnalyzing: {place_name}")
        print(f"   Emotion vector: {emotion_vector}")
        
        # Skip places with no emotions or only neutral
        if not emotion_vector or (len(emotion_vector) == 1 and "neutral" in emotion_vector):
            print(f"Discarding {place_name} - No meaningful emotions")
            continue
        
        # Find the dominant emotion(s) - emotions with the highest confidence
        max_confidence = max(emotion_vector.values())
        dominant_emotions = [emotion for emotion, confidence in emotion_vector.items() 
                           if confidence == max_confidence and emotion != "neutral"]
        
        print(f"   Dominant emotions: {dominant_emotions} (confidence: {max_confidence})")
        print(f"   User backend emotions: {user_backend_emotions}")
        
        # Check if ANY dominant emotion matches user emotions
        has_dominant_match = any(dom_emotion in user_backend_emotions for dom_emotion in dominant_emotions)
        
        if not has_dominant_match:
            print(f"Discarding {place_name} - No user emotions are dominant")
            continue
        
        # If neutral is dominant with high confidence, be more strict
        neutral_confidence = emotion_vector.get("neutral", 0)
        if neutral_confidence >= max_confidence and neutral_confidence > 0.7:
            print(f"Discarding {place_name} - Neutral is dominant with high confidence ({neutral_confidence})")
            continue
        
        # Calculate match score for ranking
        match_info = calculate_emotion_match_score(user_emotions, emotion_vector)
        
        # Boost score if user emotions are dominant
        dominance_boost = sum(1 for dom in dominant_emotions if dom in user_backend_emotions) * 0.3
        final_score = match_info["final_score"] + dominance_boost
        
        place_with_score = {
            **place,
            "match_info": match_info,
            "final_score": final_score,
            "dominant_emotions": dominant_emotions,
            "dominant_user_matches": [dom for dom in dominant_emotions if dom in user_backend_emotions]
        }
        scored_places.append(place_with_score)
        print(f"Keeping: {place_name} - Score: {final_score:.3f}, Dominant matches: {[dom for dom in dominant_emotions if dom in user_backend_emotions]}")
    
    print(f"\nFinal results: {len(scored_places)} places after dominant emotion filtering")
    
    # Sort by final score (descending)
    return sorted(scored_places, key=lambda x: x["final_score"], reverse=True)
@app.post("/recommend_places")
def recommend_places(user: UserPreference, top_k: int = 20):
    try:
        print(f"Context-aware recommendation request for location: {user.latitude}, {user.longitude}")
        print(f"Selected emotions: {user.emotions}")
        
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "error", "message": "No places in database. Use /fetch_places first."}
        
        # Use improved ranking system with context-aware analysis
        recommendations = rank_places_by_emotion_priority(places, user.emotions)
        
        # Limit results
        recommendations = recommendations[:top_k]
        
        # Add debug information for first few results
        debug_info = []
        for i, rec in enumerate(recommendations[:5]):
            match_info = rec.get("match_info", {})
            debug_info.append({
                "rank": i + 1,
                "place_name": rec.get("name", "Unknown"),
                "category": rec.get("category", "Unknown"),
                "context_snippet": rec.get("combined_context", "")[:100] + "..." if len(rec.get("combined_context", "")) > 100 else rec.get("combined_context", ""),
                "final_score": match_info.get("final_score", 0),
                "exact_matches": match_info.get("exact_matches", 0),
                "matched_emotions": match_info.get("matched_emotions", []),
                "all_emotions": list(rec.get("emotion_vector", {}).keys())
            })
        
        print(f"Generated {len(recommendations)} context-aware recommendations")
        print("Top 5 recommendations debug info:")
        for debug in debug_info:
            print(f"   {debug['rank']}. {debug['place_name']} (Score: {debug['final_score']}, Matches: {debug['exact_matches']})")
            print(f"      Context: {debug['context_snippet']}")
        
        # Clean up recommendations for response
        clean_recommendations = []
        for rec in recommendations:
            clean_rec = {k: v for k, v in rec.items() if k not in ["match_info"]}
            # Add summary of matching emotions for frontend
            match_info = rec.get("match_info", {})
            clean_rec["emotion_match_summary"] = {
                "score": match_info.get("final_score", 0),
                "matched_emotions": [emotion for emotion, confidence in match_info.get("matched_emotions", [])],
                "match_count": match_info.get("exact_matches", 0)
            }
            clean_recommendations.append(clean_rec)
        
        return {
            "status": "success",
            "user_location": {"lat": user.latitude, "lon": user.longitude},
            "user_emotions": user.emotions,
            "recommendations": clean_recommendations,
            "count": len(clean_recommendations),
            "analysis_type": "context-aware",
            "debug_info": debug_info
        }
    except Exception as e:
        print(f"Error in recommend_places: {e}")
        return {"status": "error", "message": str(e)}

# -------------------------------
# 10. Legacy functions (kept for compatibility)
# -------------------------------
def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Legacy function kept for compatibility - not used in new recommendation system"""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in common)
    norm1 = sqrt(sum(v**2 for v in vec1.values()))
    norm2 = sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def create_emotion_vector_from_emotions(selected_emotions: List[str]) -> Dict[str, float]:
    """Legacy function kept for compatibility - not used in new recommendation system"""
    emotion_mapping = {
    "Joy/Happy": ["joy, happy", "joy", "happy", "entertainment"],
    "Relaxation/Calm": ["relaxation,calm", "relaxation", "calm", "comfort", "wellness"],
    "Social Emotion": ["social", "food", "entertainment"],
    "Excitement": ["excitement", "adventure", "entertainment"],
    "Comfort": ["comfort", "wellness"],
    "Adventure": ["adventure", "excitement", "energy"],
    "Romance": ["luxury", "comfort"],  # No direct romance emotion found
    "Luxury": ["luxury", "comfort"],
    "Shopping": ["shopping", "retail", "convenience"],
    "Nostalgia": ["comfort"],  # No direct nostalgia emotion found
    "Energy": ["energy", "excitement", "adventure","wellness"],
    "Wellness": ["wellness", "healthcare"],
    "Entertainment": ["entertainment", "excitement", "joy, happy"],
    "Exploration": ["adventure", "education", "excitement"],
    "Fear": ["fear", "stress"],
    "Creativity": ["education", "entertainment"],  # No direct creativity emotion found
    "Spirituality": ["spirituality","relaxation,calm"],
    "Education": ["education", "professional"],
    "Retail": ["retail", "shopping", "convenience"],
    "Outdoors": ["adventure", "energy", "excitement"]
}

    
    vector = {}
    base_weight = 1.0 / len(selected_emotions) if selected_emotions else 0
    
    for emotion in selected_emotions:
        mapped_emotions = emotion_mapping.get(emotion, [emotion.lower()])
        for mapped_emotion in mapped_emotions:
            vector[mapped_emotion] = base_weight
    
    return vector

# -------------------------------
# 11. Additional Context-Aware Testing Endpoints
# -------------------------------
@app.get("/analyze_context/{place_name}")
def analyze_place_context(place_name: str, category: str = "restaurant", 
                          description: str = "", lat: float = None, lon: float = None):
    """
    Analyze the contextual understanding of a place
    """
    try:
        # Create context-aware text
        context_text = emotion_analyzer.create_context_aware_text(place_name, category, description)
        
        # Get emotions
        emotions = emotion_analyzer.predict_emotions_for_place(place_name, category, description)
        
        # If reference data is available, show similarity analysis
        similarity_info = None
        if emotion_analyzer.reference_data is not None:
            place_embedding = emotion_analyzer.model.encode(context_text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(place_embedding, emotion_analyzer.reference_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=5)
            
            similar_places = []
            for score, idx in zip(top_results.values, top_results.indices):
                ref_place = emotion_analyzer.reference_data.iloc[idx.item()]
                similar_places.append({
                    "reference_text": ref_place['text_for_embedding'],
                    "emotion": ref_place['place_emotion'],
                    "similarity_score": round(float(score.item()), 4)
                })
            
            similarity_info = {
                "similar_reference_places": similar_places,
                "avg_similarity": round(float(cosine_scores.mean().item()), 4)
            }
        
        return {
            "place_name": place_name,
            "category": category,
            "original_description": description,
            "context_aware_text": context_text,
            "predicted_emotions": emotions,
            "similarity_analysis": similarity_info,
            "analysis_method": "Context-aware transformer analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Add this endpoint to your main.py

@app.get("/unique_emotions")
def get_unique_emotions():
    """
    Get all unique emotions currently classified by the backend
    """
    try:
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "error", "message": "No places found"}
        
        # Collect all unique emotions
        unique_emotions = set()
        for place in places:
            for emotion_data in place.get("emotions", []):
                emotion_name = emotion_data.get("emotion")
                if emotion_name:
                    unique_emotions.add(emotion_name)
        
        # Sort alphabetically for easier reading
        sorted_emotions = sorted(list(unique_emotions))
        
        return {
            "status": "success",
            "total_unique_emotions": len(sorted_emotions),
            "unique_emotions": sorted_emotions
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.get("/context_stats")
def get_context_analysis_stats():
    """
    Get statistics about the context-aware analysis
    """
    try:
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "error", "message": "No places found"}
        
        stats = {
            "total_places": len(places),
            "places_with_descriptions": sum(1 for p in places if p.get("description", "").strip()),
            "places_with_context": sum(1 for p in places if p.get("combined_context", "").strip()),
            "avg_context_length": sum(len(p.get("combined_context", "")) for p in places) / len(places),
            "unique_categories": len(set(p.get("category", "unknown") for p in places)),
            "emotion_distribution": {},
            "context_quality_indicators": {
                "has_detailed_tags": sum(1 for p in places if len(p.get("osm_tags", {})) > 3),
                "has_cuisine_info": sum(1 for p in places if p.get("osm_tags", {}).get("cuisine")),
                "has_opening_hours": sum(1 for p in places if p.get("osm_tags", {}).get("opening_hours")),
                "has_accessibility_info": sum(1 for p in places if p.get("osm_tags", {}).get("wheelchair"))
            }
        }
        
        # Calculate emotion distribution
        all_emotions = {}
        for place in places:
            for emotion_data in place.get("emotions", []):
                emotion = emotion_data.get("emotion")
                if emotion in all_emotions:
                    all_emotions[emotion] += 1
                else:
                    all_emotions[emotion] = 1
        
        stats["emotion_distribution"] = dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "status": "success",
            "analysis_stats": stats,
            "model_info": {
                "model_name": "all-mpnet-base-v2",
                "context_aware": True,
                "reference_data_available": emotion_analyzer.reference_data is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 12. Serve frontend - Modified for Render deployment
# -------------------------------
# Check if frontend directory exists and mount it
FRONTEND_DIR = os.path.join(os.getcwd(), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    print(f"Frontend mounted from: {FRONTEND_DIR}")
else:
    print("Frontend directory not found - API only mode")

@app.get("/")
def root():
    if os.path.exists(FRONTEND_DIR):
        return RedirectResponse("/frontend/frontpage.html")
    else:
        return {
            "message": "Enhanced Places Emotion Recommender API with Context-Aware Analysis and Location Caching",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "fetch_places": "/fetch_places (POST)",
                "places": "/places",
                "recommend_places": "/recommend_places (POST)",
                "test_emotion": "/test_emotion/{place_name}",
                "debug_place": "/debug_place/{place_name}",
                "analyze_context": "/analyze_context/{place_name}",
                "context_stats": "/context_stats",
                "cache_status": "/cache_status",
                "clear_cache": "/clear_cache (POST)"
            }
        }

# -------------------------------
# 13. Modified for Render - Don't auto-open browser in production
# -------------------------------
def open_frontend():
    """Only open browser in local development"""
    if os.getenv("RENDER") != "true":  # Only open locally, not on Render
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:8000/")

# -------------------------------
# 14. Run app - Modified for Render deployment
# -------------------------------
if __name__ == "__main__":
    print("Starting Enhanced Places Emotion Recommender System...")
    print("Features: Context-aware analysis with Location Caching")
    
    # Don't start browser thread in production
    if os.getenv("RENDER") != "true":
        threading.Thread(target=open_frontend, daemon=True).start()
    
    # Get port from environment (required for Render)
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)