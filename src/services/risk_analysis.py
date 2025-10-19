def analyze_risk(disease, weather_data):
    temperature = weather_data["temperature"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]

    if disease == "Healthy":
        return "Low"

    risk_score = 0
    if disease == "Yellow":
        if humidity > 80: risk_score += 2
        if 25 <= temperature <= 35: risk_score += 1
    elif disease == "Mosaic":
        if humidity > 75: risk_score += 2
        if rainfall > 5: risk_score += 2
    elif disease == "Rust":
        if humidity > 70: risk_score += 2
        if temperature > 30: risk_score += 1
    elif disease == "Redrot":
        if temperature > 32: risk_score += 2
        if rainfall > 3: risk_score += 1

    if risk_score >= 3:
        return "High"
    elif risk_score >= 1:
        return "Medium"
    else:
        return "Low"
