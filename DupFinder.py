import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import imagehash
from io import BytesIO
import re
from urllib.parse import urlparse

# ===== Load sentence embedding model =====
model = SentenceTransformer('all-MiniLM-L6-v2')

# ===== Helper Functions =====
def get_instagram_data(username):
    """Fetch public data from an Instagram profile page (no login)."""
    url = f"https://www.instagram.com/{username}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Extract metadata
        desc_tag = soup.find("meta", attrs={"name": "description"})
        desc = desc_tag["content"] if desc_tag else ""
        
        # Extract JSON data (public only)
        scripts = soup.find_all("script", type="application/ld+json")
        json_data = scripts[0].string if scripts else ""
        
        # Extract external links in bio
        links = re.findall(r'(https?://[^\s"]+)', r.text)
        
        # Extract hashtags
        hashtags = re.findall(r"#(\w+)", desc)
        
        # Extract profile image
        img_tag = soup.find("meta", property="og:image")
        profile_img_url = img_tag["content"] if img_tag else ""
        
        return {
            "bio": desc,
            "hashtags": hashtags,
            "links": links,
            "profile_img_url": profile_img_url,
            "json_data": json_data
        }
    except Exception as e:
        print(f"Error fetching {username}: {e}")
        return {}

def embedding_similarity(text1, text2):
    """Compare bios or captions using semantic embeddings."""
    if not text1 or not text2:
        return 0
    emb1 = model.encode(text1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = model.encode(text2, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb1, emb2)) * 100

def hashtag_similarity(tags1, tags2):
    """Compute overlap percentage between two hashtag lists."""
    if not tags1 or not tags2:
        return 0
    s1, s2 = set(tags1), set(tags2)
    return len(s1 & s2) / len(s1 | s2) * 100

def link_similarity(links1, links2):
    """Compare base domains of external links."""
    domains1 = {urlparse(l).netloc for l in links1}
    domains2 = {urlparse(l).netloc for l in links2}
    if not domains1 or not domains2:
        return 0
    return len(domains1 & domains2) / len(domains1 | domains2) * 100

def image_similarity(url1, url2):
    """Compare public profile pictures using perceptual hash."""
    try:
        img1 = Image.open(BytesIO(requests.get(url1, timeout=10).content))
        img2 = Image.open(BytesIO(requests.get(url2, timeout=10).content))
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        return (1 - (hash1 - hash2) / len(hash1.hash) ** 2) * 100
    except:
        return 0

def compare_accounts(user1, user2):
    """Main comparison function."""
    print(f"\n🔍 Comparing {user1} vs {user2} ...\n")
    a = get_instagram_data(user1)
    b = get_instagram_data(user2)

    bio_sim = embedding_similarity(a.get("bio"), b.get("bio"))
    tag_sim = hashtag_similarity(a.get("hashtags"), b.get("hashtags"))
    link_sim = link_similarity(a.get("links"), b.get("links"))
    img_sim = image_similarity(a.get("profile_img_url"), b.get("profile_img_url"))

    final_score = (bio_sim * 0.4) + (tag_sim * 0.2) + (link_sim * 0.2) + (img_sim * 0.2)

    print(f"🧠 Bio Similarity: {bio_sim:.2f}%")
    print(f"#️⃣ Hashtag Overlap: {tag_sim:.2f}%")
    print(f"🔗 External Links Similarity: {link_sim:.2f}%")
    print(f"🖼️ Profile Image Similarity: {img_sim:.2f}%")
    print(f"\n✅ Estimated Connection Likelihood: {final_score:.2f}%\n")
    input("...")

# ===== Example Usage =====
if __name__ == "__main__":
    user1 = input("Enter first Instagram username (without @): ").strip()
    user2 = input("Enter second Instagram username (without @): ").strip()
    compare_accounts(user1, user2)
