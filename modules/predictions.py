import joblib
import numpy as np
import pandas as pd
from config import FEELS_LIKE_MODEL, ACTIVITY_MODEL, HEALTH_RISK_MODEL

FEATURE_COLS = [
    "feature_temp",
    "feature_humidity",
    "feature_wind",
    "feature_aqi",
    "feature_age",
    "feature_health_score",
    "feature_heat_sens",
    "feature_cold_sens",
]

# WHO and EPA health thresholds used for personalised risk assessment
# Source: WHO Environmental Health Guidelines (2023)
# Source: EPA Air Quality Index Health Impact Standards (2024)
THRESHOLDS = {
    "heat_risk_normal":        33,  # Safe heat limit for healthy adults
    "heat_risk_elderly":       30,  # Lower limit for age 65+
    "heat_risk_heart":         28,  # Lower limit for heart condition patients
    "heat_risk_asthma":        30,  # Lower limit for asthma patients
    "cold_risk_normal":         5,  # Safe cold limit for healthy adults
    "cold_risk_elderly":       10,  # Higher cold limit for age 65+
    "cold_risk_arthritis":     12,  # Cold triggers joint pain for arthritis
    "aqi_moderate":             2,  # AQI starts affecting sensitive groups
    "aqi_unhealthy_sensitive":  3,  # AQI dangerous for respiratory conditions
    "aqi_unhealthy":            4,  # AQI unhealthy for everyone
    "humidity_discomfort":     70,  # Humidity causes discomfort above this
    "humidity_heat_stress":    80,  # Humidity amplifies heat stress above this
    "uv_moderate":              3,  # UV starts causing skin damage
    "uv_high":                  6,  # UV requires sunscreen
    "uv_very_high":             8,  # UV causes damage within minutes
}


def build_runtime_features(weather, profile):
    """
    Build the 8-feature vector that matches exactly what models were trained on.
    This consistency between training and runtime is critical for valid ML predictions.
    Features: temperature, humidity, wind, aqi, age, health_score, heat_sens, cold_sens
    """
    temp       = weather["temperature"]
    humidity   = weather["humidity"]
    wind       = weather["wind_speed"]
    aqi        = weather.get("aqi", 1)
    age        = profile["age"]
    conditions = profile["health_conditions"]
    heat_sens  = profile["heat_sensitivity"]
    cold_sens  = profile["cold_sensitivity"]

    # Convert health conditions to numeric score for ML input
    health_scores = {
        "None": 0, "Asthma": 3, "Heart condition": 4,
        "Diabetes": 2, "Arthritis": 2, "Respiratory illness": 3
    }
    health_score = sum(health_scores.get(c, 0) for c in conditions)

    # Convert sensitivity labels to numeric 1-3 scale
    heat_sens_score = {"Low": 1, "Medium": 2, "High": 3}.get(heat_sens, 2)
    cold_sens_score = {"Low": 1, "Medium": 2, "High": 3}.get(cold_sens, 2)

    return [temp, humidity, wind, aqi, age, health_score, heat_sens_score, cold_sens_score]


def build_runtime_feature_frame(weather, profile):
    """Return runtime features with the same column names used during training."""
    return pd.DataFrame([build_runtime_features(weather, profile)], columns=FEATURE_COLS)


def calculate_personalisation_offset(weather, profile):
    """
    Apply WHO and research-based offset on top of ML base prediction.
    This accounts for individual differences in thermal perception.
    Source: Luo et al. (2022), Kim et al. (2021), WHO Guidelines (2023)
    """
    temp       = weather["temperature"]
    humidity   = weather["humidity"]
    age        = profile["age"]
    heat_sens  = profile["heat_sensitivity"]
    cold_sens  = profile["cold_sensitivity"]
    conditions = profile["health_conditions"]
    offset = 0.0

    # Age adjustment - elderly perceive heat 2-4 degrees higher (Kim et al., 2021)
    if age >= 65:
        if temp > 25:   offset += 2.5
        elif temp < 10: offset -= 2.0
    elif age >= 50:
        if temp > 25:   offset += 1.0
        elif temp < 10: offset -= 1.0

    # Heat sensitivity adjustment
    if heat_sens == "High" and temp > 20:     offset += 2.0
    elif heat_sens == "Medium" and temp > 20: offset += 0.8
    elif heat_sens == "Low" and temp > 20:    offset -= 0.5

    # Cold sensitivity adjustment
    if cold_sens == "High" and temp < 15:     offset -= 2.0
    elif cold_sens == "Medium" and temp < 15: offset -= 0.8
    elif cold_sens == "Low" and temp < 15:    offset += 0.5

    # Health condition adjustments based on clinical research
    if "Heart condition" in conditions and temp > 25: offset += 1.5
    if "Asthma" in conditions and humidity > 70:      offset += 1.2
    if "Respiratory illness" in conditions and humidity > 65: offset += 0.8
    if "Diabetes" in conditions and temp > 30:        offset += 0.8
    if "Arthritis" in conditions and temp < 15:       offset -= 1.2

    # Humidity amplifies heat perception significantly above 80%
    if humidity > 80 and temp > 25:   offset += 1.2
    elif humidity > 70 and temp > 28: offset += 0.8

    return round(offset, 1)


def predict_feels_like(weather, profile):
    """
    Predict personalised feels-like temperature.
    Uses Random Forest Regressor trained on ASHRAE data + WHO personalisation offset.
    Two-layer approach: ML base prediction + research-based personal adjustment.
    """
    try:
        # Load trained Random Forest Regressor
        model = joblib.load(FEELS_LIKE_MODEL)

        # Build feature vector matching training features exactly
        features = build_runtime_feature_frame(weather, profile)

        # Get ML base prediction from model
        base_prediction = float(model.predict(features)[0])

        # Add WHO-based personalisation offset on top
        offset = calculate_personalisation_offset(weather, profile)

        return round(base_prediction + offset, 1)

    except Exception:
        # Fallback to API feels-like + offset if model fails
        offset = calculate_personalisation_offset(weather, profile)
        return round(weather["feels_like_api"] + offset, 1)


def predict_health_risk(weather, profile):
    """
    Predict personalised health risk using two-layer approach:
    Layer 1: Trained ML model (Neural Network selected for health-risk deployment)
    Layer 2: WHO and EPA clinical threshold validation
    Combined result takes the maximum of both for safety.
    Returns: risk_level, explanation, confidence, risk_factors list
    """
    temp       = weather["temperature"]
    humidity   = weather["humidity"]
    aqi        = weather.get("aqi", 1)
    uv         = weather.get("uv_index", 0)
    age        = profile["age"]
    conditions = profile["health_conditions"]
    heat_sens  = profile["heat_sensitivity"]
    cold_sens  = profile["cold_sensitivity"]

    # Layer 1: Get ML model prediction
    ml_risk = None
    ml_confidence = None
    try:
        model = joblib.load(HEALTH_RISK_MODEL)
        features = build_runtime_feature_frame(weather, profile)

        # Try Neural Network path with StandardScaler first
        try:
            scaler = joblib.load("models/scaler.pkl")
            features_scaled = scaler.transform(features)
            ml_pred = model.predict(features_scaled)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                ml_confidence = round(float(max(proba)) * 100, 1)
        except Exception:
            # Fallback for non-neural network models
            ml_pred = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                ml_confidence = round(float(max(proba)) * 100, 1)

        ml_risk = int(ml_pred)
    except Exception:
        ml_risk = None

    # Layer 2: WHO and EPA rule-based risk assessment
    risk_factors = []
    risk_score   = 0

    # Set personalised thresholds based on health profile
    if age >= 65:
        heat_threshold = THRESHOLDS["heat_risk_elderly"]
        cold_threshold = THRESHOLDS["cold_risk_elderly"]
    elif "Heart condition" in conditions:
        heat_threshold = THRESHOLDS["heat_risk_heart"]
        cold_threshold = THRESHOLDS["cold_risk_normal"]
    elif "Asthma" in conditions or "Respiratory illness" in conditions:
        heat_threshold = THRESHOLDS["heat_risk_asthma"]
        cold_threshold = THRESHOLDS["cold_risk_normal"]
    else:
        heat_threshold = THRESHOLDS["heat_risk_normal"]
        cold_threshold = THRESHOLDS["cold_risk_normal"]

    # Temperature risk assessment
    if temp > heat_threshold + 5:
        risk_score += 4
        risk_factors.append({"factor": "Extreme Heat",
            "detail": f"Temperature {temp}C is {round(temp-heat_threshold,1)}C above your safe threshold of {heat_threshold}C",
            "severity": "High"})
    elif temp > heat_threshold:
        risk_score += 2
        risk_factors.append({"factor": "High Temperature",
            "detail": f"Temperature {temp}C exceeds your personal safe threshold of {heat_threshold}C",
            "severity": "Medium"})

    if temp < cold_threshold - 5:
        risk_score += 4
        risk_factors.append({"factor": "Extreme Cold",
            "detail": f"Temperature {temp}C is significantly below your safe threshold of {cold_threshold}C",
            "severity": "High"})
    elif temp < cold_threshold:
        risk_score += 2
        risk_factors.append({"factor": "Low Temperature",
            "detail": f"Temperature {temp}C is below your personal safe threshold of {cold_threshold}C",
            "severity": "Medium"})

    # Air quality risk - stricter thresholds for respiratory conditions
    if "Asthma" in conditions or "Respiratory illness" in conditions:
        if aqi >= THRESHOLDS["aqi_unhealthy_sensitive"]:
            risk_score += 4
            risk_factors.append({"factor": "Dangerous Air Quality",
                "detail": f"AQI {aqi} is dangerous for your respiratory condition. Outdoor exposure should be avoided",
                "severity": "High"})
        elif aqi >= THRESHOLDS["aqi_moderate"]:
            risk_score += 2
            risk_factors.append({"factor": "Poor Air Quality",
                "detail": f"AQI {aqi} may trigger respiratory symptoms. Limit outdoor exposure",
                "severity": "Medium"})
    else:
        if aqi >= THRESHOLDS["aqi_unhealthy"]:
            risk_score += 3
            risk_factors.append({"factor": "Very Poor Air Quality",
                "detail": f"AQI {aqi} is unhealthy for all individuals",
                "severity": "High"})
        elif aqi >= THRESHOLDS["aqi_unhealthy_sensitive"]:
            risk_score += 1
            risk_factors.append({"factor": "Moderate Air Quality",
                "detail": f"AQI {aqi} may affect sensitive individuals",
                "severity": "Low"})

    # Humidity risk - amplifies heat stress significantly
    if humidity > THRESHOLDS["humidity_heat_stress"] and temp > 25:
        risk_score += 3
        risk_factors.append({"factor": "High Humidity Heat Stress",
            "detail": f"Humidity {humidity}% combined with {temp}C creates significant heat stress",
            "severity": "High"})
    elif humidity > THRESHOLDS["humidity_discomfort"]:
        risk_score += 1
        risk_factors.append({"factor": "Elevated Humidity",
            "detail": f"Humidity {humidity}% may cause discomfort during physical activity",
            "severity": "Low"})

    # UV radiation risk assessment
    if uv >= THRESHOLDS["uv_very_high"]:
        risk_score += 3
        risk_factors.append({"factor": "Very High UV Radiation",
            "detail": f"UV Index {uv} is at very high level. Skin damage can occur within minutes",
            "severity": "High"})
    elif uv >= THRESHOLDS["uv_high"]:
        risk_score += 2
        risk_factors.append({"factor": "High UV Radiation",
            "detail": f"UV Index {uv} requires sunscreen and limited sun exposure",
            "severity": "Medium"})

    # Heart condition specific risk
    if "Heart condition" in conditions:
        if temp > 28 or temp < 5:
            risk_score += 3
            risk_factors.append({"factor": "Cardiovascular Risk",
                "detail": f"Temperature {temp}C places significant stress on the cardiovascular system",
                "severity": "High"})
        else:
            risk_score += 2
            risk_factors.append({"factor": "Cardiovascular Caution",
                "detail": "Heart condition patients require careful monitoring in all weather conditions",
                "severity": "Medium"})

    # Asthma baseline risk always present outdoors
    if "Asthma" in conditions:
        risk_score += 1
        risk_factors.append({"factor": "Asthma Sensitivity",
            "detail": "Asthma sufferers have increased sensitivity to outdoor conditions",
            "severity": "Low"})

    # Diabetes heat risk
    if "Diabetes" in conditions and temp > 30:
        risk_score += 2
        risk_factors.append({"factor": "Diabetes Heat Risk",
            "detail": f"High temperature {temp}C affects insulin absorption and blood sugar regulation",
            "severity": "Medium"})

    # Arthritis cold risk
    if "Arthritis" in conditions and temp < 12:
        risk_score += 2
        risk_factors.append({"factor": "Arthritis Cold Risk",
            "detail": f"Temperature {temp}C may increase joint stiffness and pain",
            "severity": "Medium"})

    # Age related vulnerability added when other risks present
    if age >= 65 and risk_score > 0:
        risk_score += 2
        risk_factors.append({"factor": "Age Related Vulnerability",
            "detail": f"At age {age}, thermoregulation is less efficient, increasing sensitivity to all weather risks",
            "severity": "Medium"})

    # Personal sensitivity adjustments
    if heat_sens == "High" and temp > 25:
        risk_score += 1
        risk_factors.append({"factor": "High Heat Sensitivity",
            "detail": "Your heat sensitivity profile increases risk in current warm conditions",
            "severity": "Low"})

    if cold_sens == "High" and temp < 15:
        risk_score += 1
        risk_factors.append({"factor": "High Cold Sensitivity",
            "detail": "Your cold sensitivity profile increases risk in current cool conditions",
            "severity": "Low"})

    # Convert rule score to risk label (0=Low, 1=Medium, 2=High)
    rule_risk = 2 if risk_score >= 8 else (1 if risk_score >= 4 else 0)
    labels = {0: "Low", 1: "Medium", 2: "High"}

    # Combine ML and rule-based predictions - take maximum for safety
    if ml_risk is not None:
        combined_risk = max(ml_risk, rule_risk)
        risk_level = labels[combined_risk]
        confidence = ml_confidence if ml_confidence else (
            min(95, 75 + risk_score * 2) if combined_risk == 2 else
            min(90, 65 + risk_score * 3) if combined_risk == 1 else
            min(95, 80 + (3 - risk_score) * 5))
    else:
        risk_level = labels[rule_risk]
        confidence = (
            min(95, 75 + risk_score * 2) if rule_risk == 2 else
            min(90, 65 + risk_score * 3) if rule_risk == 1 else
            min(95, 80 + (3 - risk_score) * 5))

    # Generate human readable explanation
    if not risk_factors:
        explanation = (
            f"Current conditions are within safe parameters. "
            f"Temperature {temp}C, humidity {humidity}% and AQI {aqi} "
            f"are all within acceptable ranges for your profile.")
    else:
        high_factors   = [f for f in risk_factors if f["severity"] == "High"]
        medium_factors = [f for f in risk_factors if f["severity"] == "Medium"]
        if high_factors:
            explanation = f"{risk_level} risk identified. Primary concerns: {', '.join([f['factor'] for f in high_factors])}."
        elif medium_factors:
            explanation = f"{risk_level} risk identified. Moderate concerns: {', '.join([f['factor'] for f in medium_factors])}."
        else:
            explanation = f"{risk_level} risk identified. Minor concerns: {', '.join([f['factor'] for f in risk_factors])}."

    return risk_level, explanation, round(confidence, 1), risk_factors


def _activity_percentage(score):
    """Convert the internal activity score to a user-friendly suitability percentage."""
    return int(max(0, min(100, round(((score + 5) / 13) * 100))))


def _activity_safety_precautions(activity, status, reasons_bad, conditions):
    precautions = []

    if activity in ["Running", "Cycling"]:
        precautions.extend([
            "warm up slowly",
            "keep intensity moderate",
            "carry water",
        ])
    elif activity == "Walking":
        precautions.extend([
            "choose a steady pace",
            "wear comfortable shoes",
        ])
    elif activity == "Gardening":
        precautions.extend([
            "take shade breaks",
            "wear gloves and sun protection",
        ])
    elif activity == "Swimming":
        precautions.extend([
            "avoid swimming alone",
            "check water temperature first",
        ])

    if "Asthma" in conditions or "Respiratory illness" in conditions:
        precautions.extend([
            "carry your inhaler or prescribed medication",
            "avoid high intensity effort if breathing becomes difficult",
        ])
    if "Heart condition" in conditions:
        precautions.extend([
            "avoid sudden high exertion",
            "stop immediately if you feel chest pain, dizziness, or unusual breathlessness",
        ])
    if "Diabetes" in conditions:
        precautions.extend([
            "carry a quick sugar source",
            "monitor hydration and energy levels",
        ])
    if "Arthritis" in conditions:
        precautions.extend([
            "protect affected joints",
            "choose low impact movement where possible",
        ])

    if reasons_bad:
        joined = "; ".join(reasons_bad[:3])
        precaution_text = ", ".join(dict.fromkeys(precautions[:5]))
        return (
            f"To do {activity.lower()} more safely, {precaution_text}. "
            f"Be careful because current limiting factors include: {joined}."
        )

    if status == "Ideal":
        precaution_text = ", ".join(dict.fromkeys(precautions[:4])) or "stay hydrated and stop if symptoms appear"
        return (
            f"{activity} is currently a strong option. For safety, {precaution_text}."
        )

    if any(c in conditions for c in ["Asthma", "Respiratory illness"]):
        return (
            f"If you do {activity.lower()} anyway, watch for breathing difficulty, wheezing, "
            "or chest tightness because your respiratory profile increases outdoor sensitivity."
        )

    if "Heart condition" in conditions:
        return (
            f"If you do {activity.lower()} anyway, avoid high intensity effort because it may "
            "increase cardiovascular load."
        )

    return (
        f"To do {activity.lower()} more safely, keep the session short, reduce intensity, "
        "stay hydrated, and stop if symptoms appear."
    )


def predict_activities(weather, profile):
    """
    Predict activity suitability for each user preference.
    Three-layer scoring system:
    Layer 1: ML model base score (Random Forest trained on ASHRAE)
    Layer 2: Weather condition scoring (temperature, humidity, AQI, UV, wind)
    Layer 3: Health condition and age penalties/bonuses
    Final classification: Ideal (score>=4), Suitable (score>=1), Not Recommended (score<1)
    """
    temp       = weather["temperature"]
    humidity   = weather["humidity"]
    wind       = weather["wind_speed"]
    aqi        = weather.get("aqi", 1)
    uv         = weather.get("uv_index", 0)
    conditions = profile["health_conditions"]
    age        = profile["age"]

    # Only assess activities the user selected in their profile
    activities = [a for a in profile.get("activity_preferences", []) if a != "None"]

    # Load trained activity classification model
    try:
        act_model = joblib.load(ACTIVITY_MODEL)
        use_ml = True
    except Exception:
        use_ml = False

    results = {}

    for activity in activities:
        score        = 0
        reasons_good = []
        reasons_bad  = []

        # Layer 1: ML model base prediction
        if use_ml:
            try:
                features = build_runtime_feature_frame(weather, profile)
                ml_pred  = act_model.predict(features)[0]
                # Map ML output to score: 0=Ideal(+3), 1=Suitable(+1), 2=NotRecommended(-2)
                if ml_pred == 0:   score += 3
                elif ml_pred == 1: score += 1
                else:              score -= 2
            except Exception:
                pass

        # Layer 2: Weather condition scoring
        if 15 <= temp <= 25:
            score += 2
            reasons_good.append(f"temperature {temp}C is ideal for outdoor activity")
        elif 10 <= temp <= 30:
            score += 1
            reasons_good.append(f"temperature {temp}C is within acceptable range")
        elif temp > 35:
            score -= 4
            reasons_bad.append(f"temperature {temp}C is dangerously high for physical activity")
        elif temp > 30:
            score -= 2
            reasons_bad.append(f"temperature {temp}C is too hot for safe outdoor exercise")
        elif temp < 0:
            score -= 4
            reasons_bad.append(f"temperature {temp}C is dangerously cold for outdoor activity")
        elif temp < 5:
            score -= 2
            reasons_bad.append(f"temperature {temp}C is very cold for outdoor activity")

        if humidity < 60:
            score += 1
            reasons_good.append("humidity levels are comfortable")
        elif humidity > 85:
            score -= 2
            reasons_bad.append(f"humidity at {humidity}% makes breathing very difficult")
        elif humidity > 75:
            score -= 1
            reasons_bad.append(f"humidity at {humidity}% increases physical effort significantly")

        if aqi <= 2:
            score += 1
            reasons_good.append("air quality is good")
        elif aqi == 3:
            score -= 1
            reasons_bad.append("moderate air quality may affect breathing during exercise")
        elif aqi >= 4:
            score -= 3
            reasons_bad.append(f"air quality index of {aqi} is unhealthy for outdoor exercise")

        # Activity specific weather adjustments
        if activity == "Running":
            if wind > 10:
                score -= 2
                reasons_bad.append(f"wind speed of {wind}m/s makes running very difficult")
            if uv >= 6:
                score -= 2
                reasons_bad.append(f"UV index of {uv} is too high for prolonged running")
            if temp > 28:
                score -= 2
                reasons_bad.append(f"running in {temp}C heat significantly increases heat exhaustion risk")
        elif activity == "Cycling":
            if wind > 8:
                score -= 2
                reasons_bad.append(f"wind speed of {wind}m/s creates significant resistance for cycling")
            if temp > 30:
                score -= 1
                reasons_bad.append("cycling in this heat significantly increases dehydration risk")
        elif activity == "Walking":
            if wind > 15:
                score -= 1
                reasons_bad.append(f"wind speed of {wind}m/s makes walking uncomfortable")
        elif activity == "Gardening":
            if uv >= 6:
                score -= 2
                reasons_bad.append(f"UV index of {uv} is dangerous during prolonged sun exposure")
            if temp > 32:
                score -= 1
                reasons_bad.append("gardening in this heat increases sunstroke risk")
        elif activity == "Swimming":
            if temp < 15:
                score -= 2
                reasons_bad.append(f"temperature {temp}C is too cold for outdoor swimming")
            if uv >= 8:
                score -= 1
                reasons_bad.append(f"UV index of {uv} is very high. High SPF sunscreen is essential")

        # Layer 3: Health condition penalties and bonuses
        # Asthma - high intensity cardio always risky
        if "Asthma" in conditions:
            if activity in ["Running", "Cycling"]:
                score -= 2
                reasons_bad.append(f"high intensity cardio raises breathing rate which is particularly risky for asthma sufferers")
            if aqi >= 3:
                score -= 3
                reasons_bad.append(f"air quality index of {aqi} poses serious risk for asthma sufferers outdoors")
            if humidity > 75:
                score -= 2
                reasons_bad.append("elevated humidity can trigger asthma symptoms during physical activity")
            if aqi >= 2 and activity in ["Running", "Cycling"]:
                score -= 1
                reasons_bad.append("even moderate air quality increases asthma risk during vigorous exercise")

        # Heart condition - strenuous activities always penalised
        if "Heart condition" in conditions:
            if activity == "Running":
                score -= 3
                reasons_bad.append("running creates high cardiovascular demand which carries serious risk for heart condition patients")
            elif activity == "Cycling":
                score -= 2
                reasons_bad.append("sustained cycling increases cardiac workload which is risky for heart condition patients")
            if temp > 28:
                score -= 2
                reasons_bad.append(f"temperature {temp}C places additional stress on the cardiovascular system")
            elif temp < 5:
                score -= 2
                reasons_bad.append("cold temperatures cause blood vessel constriction which increases cardiovascular risk")

        # Respiratory illness
        if "Respiratory illness" in conditions:
            if activity in ["Running", "Cycling"]:
                score -= 2
                reasons_bad.append("high intensity exercise significantly worsens respiratory illness symptoms")
            if aqi >= 3:
                score -= 3
                reasons_bad.append("poor air quality is particularly dangerous for respiratory illness patients")
            if temp < 5:
                score -= 2
                reasons_bad.append("cold air aggravates respiratory conditions")

        # Diabetes - exercise beneficial but heat risky
        if "Diabetes" in conditions:
            if temp > 30:
                score -= 2
                reasons_bad.append("extreme heat disrupts blood sugar regulation in diabetes patients")
            if activity in ["Running", "Cycling"]:
                score += 1
                reasons_good.append("moderate exercise supports diabetes management")

        # Arthritis - running always risky, swimming always beneficial
        if "Arthritis" in conditions:
            if activity == "Running":
                score -= 2
                reasons_bad.append("high impact running increases joint stress for arthritis patients")
            if activity == "Swimming":
                score += 2
                reasons_good.append("swimming is low impact and recommended for arthritis patients")
            if temp < 12:
                score -= 2
                reasons_bad.append(f"cold temperature of {temp}C increases joint stiffness and pain")

        # Age 65+ adjustments - walking ideal, running always penalised
        if age >= 65:
            if activity == "Running":
                score -= 2
                reasons_bad.append(f"running carries elevated cardiovascular risk at age {age}")
            if activity == "Walking":
                score += 2
                reasons_good.append(f"walking is the ideal low-impact activity at age {age}")
            if activity == "Swimming":
                score += 1
                reasons_good.append(f"swimming provides excellent gentle exercise at age {age}")
            if temp > 30:
                score -= 2
                reasons_bad.append(f"heat regulation declines with age making {temp}C significantly more dangerous at age {age}")

        # Build explanation prioritising health conditions over weather
        health_kw  = ["cardiovascular", "cardiac", "insulin", "joint",
                      "risky for asthma", "risky for heart", "worsens respiratory",
                      "demand", "raises breathing", "blood sugar"]
        age_kw     = ["age", "elderly"]
        aqi_kw     = ["air quality index", "aqi"]

        health_reasons  = [r for r in reasons_bad if any(w in r.lower() for w in health_kw)]
        age_reasons     = [r for r in reasons_bad if any(w in r.lower() for w in age_kw)]
        aqi_reasons     = [r for r in reasons_bad if any(w in r.lower() for w in aqi_kw)]
        weather_reasons = [r for r in reasons_bad if not any(
            w in r.lower() for w in health_kw + age_kw + aqi_kw)]
        good_reasons    = [r for r in reasons_good if "diabetes" not in r.lower()]

        # Generate final status and explanation
        if score >= 4:
            status = "Ideal"
            if good_reasons:
                base  = good_reasons[0][0].upper() + good_reasons[0][1:]
                extra = (". " + good_reasons[1][0].upper() + good_reasons[1][1:]) if len(good_reasons) > 1 else ""
                summary = f"Conditions are well suited for {activity.lower()} today. {base}{extra}."
            else:
                summary = f"Conditions are well suited for {activity.lower()} today."

        elif score >= 1:
            status = "Suitable"
            parts  = []
            if weather_reasons: parts.append(weather_reasons[0])
            if health_reasons:  parts.append(health_reasons[0])
            if age_reasons:     parts.append(age_reasons[0])
            if aqi_reasons and not any("aqi" in p.lower() for p in parts):
                parts.append(aqi_reasons[0])
            if parts:
                first   = parts[0][0].upper() + parts[0][1:]
                summary = f"{activity} is suitable today with some caution. {first}."
                if len(parts) > 1:
                    second  = parts[1][0].upper() + parts[1][1:]
                    summary += f" {second}."
            else:
                summary = f"Conditions are generally acceptable for {activity.lower()} today."

        else:
            status = "Not Recommended"
            parts  = []
            if health_reasons:  parts.append(health_reasons[0])
            if age_reasons:     parts.append(age_reasons[0])
            if aqi_reasons:     parts.append(aqi_reasons[0])
            if weather_reasons: parts.append(weather_reasons[0])
            if parts:
                first   = parts[0][0].upper() + parts[0][1:]
                summary = f"{activity} is not recommended today. {first}."
                if len(parts) > 1:
                    second  = parts[1][0].upper() + parts[1][1:]
                    summary += f" {second}."
            else:
                summary = f"Current conditions are not suitable for {activity.lower()} today."

        # Store result with status, explanation and score
        percentage = _activity_percentage(score)
        results[activity] = {
            "status":      status,
            "explanation": summary,
            "score":       score,
            "percentage":  percentage,
            "recommendation": (
                f"{percentage}% suitable for {activity.lower()} right now."
            ),
            "limiting_factors": reasons_bad[:4],
            "positive_factors": reasons_good[:4],
            "safety_precautions": _activity_safety_precautions(
                activity, status, reasons_bad, conditions
            )
        }

    return results


def recommend_activity_times(forecast_items, profile, current_weather=None):
    """Recommend the best forecast time windows for each selected activity."""
    activities = [a for a in profile.get("activity_preferences", []) if a != "None"]
    recommendations = {activity: [] for activity in activities}

    for item in forecast_items or []:
        forecast_weather = {
            "temperature": item["temperature"],
            "feels_like_api": item.get("feels_like_api", item["temperature"]),
            "humidity": item["humidity"],
            "wind_speed": item["wind_speed"],
            "aqi": (current_weather or {}).get("aqi", 1),
            "uv_index": 0,
        }
        slot_results = predict_activities(forecast_weather, profile)
        for activity, result in slot_results.items():
            rain_penalty = 20 if item.get("rain_probability", 0) >= 60 else 10 if item.get("rain_probability", 0) >= 35 else 0
            percentage = max(0, result["percentage"] - rain_penalty)
            recommendations.setdefault(activity, []).append({
                "datetime": item["datetime"],
                "temperature": item["temperature"],
                "humidity": item["humidity"],
                "wind_speed": item["wind_speed"],
                "condition": item["description"],
                "rain_probability": item.get("rain_probability", 0),
                "percentage": percentage,
                "status": result["status"],
                "health_note": result["safety_precautions"],
                "activity_note": result["explanation"],
                "reason": (
                    f"{percentage}% suitable: {item['description']}, "
                    f"{item['temperature']}C, humidity {item['humidity']}%, "
                    f"wind {item['wind_speed']}m/s, rain chance {item.get('rain_probability', 0)}%."
                )
            })

    best_times = {}
    for activity, slots in recommendations.items():
        ranked = sorted(slots, key=lambda s: s["percentage"], reverse=True)
        best_times[activity] = ranked[:3]

    return best_times
