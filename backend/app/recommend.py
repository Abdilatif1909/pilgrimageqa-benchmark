from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import math

def haversine(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

class Recommender:
    def __init__(self, places: List[dict]):
        self.places = places
        corpus = [p.get('description','') + ' ' + p.get('history','') + ' ' + p.get('category','') for p in places]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf = self.vectorizer.fit_transform(corpus)

    def recommend_by_place(self, place_name: str, top_k: int = 3):
        # find place index by exact or case-insensitive match
        idx = None
        for i, p in enumerate(self.places):
            if p.get('name','').lower() == place_name.lower():
                idx = i
                break
        if idx is None:
            # try partial match
            for i, p in enumerate(self.places):
                if place_name.lower() in p.get('name','').lower():
                    idx = i
                    break
        if idx is None:
            return None

        qv = self.tfidf[idx]
        sims = linear_kernel(qv, self.tfidf).flatten()
        sims[idx] = -1  # exclude same
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            p = self.places[int(i)]
            results.append({
                'name': p.get('name'),
                'description': p.get('description'),
                'score': float(sims[int(i)])
            })
        return results

    def recommend_by_text(self, text: str, top_k: int = 3, user_location: Optional[str] = None):
        qv = self.vectorizer.transform([text])
        sims = linear_kernel(qv, self.tfidf).flatten()
        # combine with location if provided
        user_lat = user_lon = None
        if user_location:
            try:
                parts = [p.strip() for p in user_location.split(',')]
                if len(parts) >= 2:
                    user_lat = float(parts[0])
                    user_lon = float(parts[1])
            except Exception:
                user_lat = user_lon = None

        scores = []
        for i, s in enumerate(sims):
            final_score = float(s)
            distance_km = None
            if user_lat is not None and self.places[i].get('location'):
                lat = self.places[i]['location'].get('lat')
                lon = self.places[i]['location'].get('lon')
                if lat is not None and lon is not None:
                    distance_km = haversine(user_lat, user_lon, lat, lon)
                    # weight closer places slightly higher
                    final_score += max(0, (1.0 - min(distance_km / 1000.0, 1.0))) * 0.2
            scores.append((i, final_score, distance_km))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score, distance in scores[:top_k]:
            p = self.places[int(i)]
            r = {
                'name': p.get('name'),
                'description': p.get('description'),
                'score': float(score)
            }
            if distance is not None:
                r['distance_km'] = round(distance, 2)
            results.append(r)
        return results
