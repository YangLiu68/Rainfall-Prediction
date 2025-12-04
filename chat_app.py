import re
import numpy as np
import pandas as pd
import torch
import streamlit as st
import dateparser
from openai import OpenAI

from daily_dataset import DailyWeatherDataset
from train_daily_model import RainfallTransformer, get_device

MODEL_PATH = "daily_transformer_best.pt"
client = OpenAI()   # will use OPENAI_API_KEY from environment


# ============= 0. CSS for left/right chat bubbles (auto width) =============
def inject_css():
    st.markdown(
        """
        <style>
        .user-bubble {
            background-color: #DCF8C6;
            padding: 0.5rem 0.75rem;
            border-radius: 1rem;
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
            word-wrap: break-word;
            display: inline-block;
            max-width: 80%;
        }
        .bot-bubble {
            background-color: #FFFFFF;
            padding: 0.5rem 0.75rem;
            border-radius: 1rem;
            margin-bottom: 0.4rem;
            border: 1px solid #EEEEEE;
            font-size: 0.95rem;
            word-wrap: break-word;
            display: inline-block;
            max-width: 80%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============= 1. Load model & data (cached) =============
@st.cache_resource
def load_model_and_data():
    device = get_device()

    daily_ds = DailyWeatherDataset(
        csv_path="./dataset_global/weather_daily_global.csv",
        cities=("San Francisco", "New York"),
        date_col="time",
    )

    seq_len = 30
    input_dim = len(daily_ds.feature_cols)

    model = RainfallTransformer(input_dim=input_dim, seq_len=seq_len).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, device, daily_ds, seq_len


# ============= 2. Build [1, L, D] input for city+date =============
def build_input_by_city_date(daily_ds, city: str, target_date: pd.Timestamp, lookback: int):
    df = daily_ds.df
    df_city = df[df["city"] == city].sort_values("date").reset_index(drop=True)

    if target_date not in df_city["date"].values:
        raise ValueError(f"Date {target_date.date()} not found in historical data for {city}.")

    idx_list = df_city.index[df_city["date"] == target_date].tolist()
    idx = idx_list[0]

    if idx < lookback:
        raise ValueError(
            f"Not enough history before {target_date.date()} for {city} "
            f"(need {lookback} days, only have {idx})."
        )

    start = idx - lookback
    end = idx

    window = df_city.loc[start:end - 1, daily_ds.feature_cols].values.astype("float32")  # [L, D]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # [1, L, D]
    return x


# ============= 3. Heuristic mm -> probability =============
def rain_probability(pred_mm: float) -> float:
    """
    Simple heuristic mapping from predicted rainfall (mm)
    to a probability that it rains that day.
    """
    if pred_mm <= 0.05:
        p = 0.05   # almost no rain
    elif pred_mm <= 0.5:
        p = 0.30   # light chance
    elif pred_mm <= 2.0:
        p = 0.60   # moderate chance
    elif pred_mm <= 5.0:
        p = 0.80   # high chance
    else:
        p = 0.95   # very likely
    return p


# ============= 4. Parse city & date (supports SF 2020/1/12, New York March 1 2020) =============
def parse_city_and_date(text: str):
    """
    Try to extract city and date from a user query.
    """
    text_lower = text.lower()

    # 1) City detection
    city_map = {
        "san francisco": "San Francisco",
        "sf": "San Francisco",
        "new york": "New York",
        "nyc": "New York",
    }

    city = None
    city_key_matched = None
    for k, v in city_map.items():
        if k in text_lower:
            city = v
            city_key_matched = k
            break

    # 2) Date detection: 2020-01-12 / 2020/1/12
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        return city, date_str

    # 3) Fallback: use dateparser, preferably on the part after the city name
    text_for_date = text
    if city_key_matched is not None:
        idx = text_lower.find(city_key_matched)
        text_for_date = text[idx + len(city_key_matched):]

    dt = dateparser.parse(
        text_for_date.strip(),
        languages=["en"],
        settings={"PREFER_DAY_OF_MONTH": "first"},
    )

    if dt is None:
        dt = dateparser.parse(
            text,
            languages=["en"],
            settings={"PREFER_DAY_OF_MONTH": "first"},
        )

    if dt is not None:
        date_str = dt.strftime("%Y-%m-%d")
        return city, date_str

    return city, None


# ============= 5. Call LLM =============
def call_llm(user_question: str, context_text: str) -> str:
    """
    Use an LLM to answer the user.

    - You specialise in rainfall and weather.
    - If context_text contains a numeric prediction, explain it.
    - For general questions about rain / rainfall / precipitation / weather,
      give a short explanation in simple English.
    - For questions clearly unrelated to rain / rainfall / precipitation /
      weather or this rainfall model, politely refuse.
    """

    system_prompt = (
        "You are a professional meteorological assistant specializing in "
        "rainfall analysis, weather interpretation, and machine-learning–based "
        "precipitation prediction.\n\n"
        "Your responsibilities include:\n\n"
        "1. Explaining rainfall predictions:\n"
        "   When you are given numerical predictions (rainfall in mm and an "
        "   estimated rain probability) from the ML model, explain them clearly "
        "   in simple and friendly English. Give context such as what the amount "
        "   of rainfall usually means (e.g., light vs heavy rain) and what the "
        "   probability implies for the chance of rain.\n\n"
        "2. General weather and rainfall education:\n"
        "   You may answer general weather-related questions, such as:\n"
        "   - What is rain / rainfall / precipitation?\n"
        "   - What factors influence rainfall?\n"
        "   - What is humidity and how does it affect rain?\n"
        "   - What causes heavy rain, storms, or flooding?\n"
        "   - How temperature, humidity, wind, and pressure relate to weather.\n"
        "   - Differences between drizzle, showers, downpours, and thunderstorms.\n"
        "   Keep explanations short, accurate, and easy for the public to understand.\n\n"
        "3. Discussion of meteorological factors:\n"
        "   You may discuss concepts related to:\n"
        "   - humidity\n"
        "   - temperature\n"
        "   - wind speed and direction\n"
        "   - atmospheric pressure\n"
        "   - cloud cover\n"
        "   - evaporation and condensation\n"
        "   - seasonal effects on precipitation\n"
        "   - simple explanations of climate drivers (e.g., El Niño / La Niña)\n"
        "   - extreme rainfall events and flood risk\n"
        "   Always keep the explanation at a simple educational level.\n\n"
        "4. Scope of local numerical predictions:\n"
        "   The ONLY cities for which you can provide NUMERICAL day-level "
        "   rainfall predictions are San Francisco and New York, based strictly "
        "   on the ML model outputs provided in the context. If the user's "
        "   question includes one of these cities AND a specific date, use the "
        "   prediction context to answer.\n\n"
        "5. Weather questions outside the prediction scope:\n"
        "   If the user asks about weather in other cities or without a specific "
        "   date, you may still provide general explanations, but you must make "
        "   it clear that you cannot provide numerical forecasts for those places.\n\n"
        "6. Refuse unrelated questions:\n"
        "   If the user asks about topics not connected to rain, rainfall, "
        "   precipitation, weather, humidity, storms, clouds, temperature, wind, "
        "   pressure, or this rainfall model, politely refuse and say that you "
        '   are only able to talk about rainfall and weather topics.\n\n'
        "Tone:\n"
        " - Friendly, concise, factual, and easy to understand.\n"
        " - Sound like a real meteorological AI assistant on a professional "
        "   weather website."
    )

    # 把用户问题 + 模型上下文组合成一段发给 LLM
    user_content = (
        f"User question:\n{user_question}\n\n"
        f"Context from ML model (may be empty):\n{context_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 如需更换模型，在这里改名字
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()



# ============= 6. Answer one question (combine ML + LLM) =============
def answer_question(question: str, model, device, daily_ds, lookback: int) -> str:
    """
    Combine your local ML rainfall model with the LLM.

    - Try to parse city + date from the user question.
    - If successful and city is San Francisco / New York, run the ML model
      and build a short context sentence including rainfall (mm) and chance of rain.
    - Then ALWAYS call the LLM, passing in:
        - the original user question
        - the context_text (may be empty)
    - The LLM will decide whether to explain the prediction, give general
      weather info, or politely refuse unrelated questions.
    """
    city, date_str = parse_city_and_date(question)

    context_text = ""

    # 如果能解析出城市和日期，就尝试用本地模型预测
    if city is not None and date_str is not None:
        try:
            target_date = pd.Timestamp(date_str)
            x = build_input_by_city_date(daily_ds, city, target_date, lookback).to(device)

            with torch.no_grad():
                pred_log = model(x)[0].item()
                pred_mm = float(np.expm1(pred_log))

            prob = rain_probability(pred_mm) * 100.0

            context_text = (
                f"For city {city} on {target_date.date()}, "
                f"the ML model predicts about {pred_mm:.3f} mm of rainfall, "
                f"with an estimated chance of rain of roughly {prob:.0f}%."
            )
        except Exception as e:
            # 预测出错也告诉 LLM，它可以解释“模型当前无法提供预测”
            context_text = f"Could not compute a prediction because of this error: {e}"

    # 不管有没有 context_text，统一交给 LLM 根据 system prompt 决定怎么回答
    llm_reply = call_llm(question, context_text)
    return llm_reply


# ============= 7. Render history: user right, bot left (auto-width bubbles) =============
def render_history(history):
    for role, content in history:
        # escape HTML special characters, then convert newlines to <br>
        safe_text = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        safe_text = safe_text.replace("\n", "<br>")

        if role == "assistant":
            # bot bubble on the left
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-start;">
                    <div class="bot-bubble">{safe_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # user bubble on the right
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end;">
                    <div class="user-bubble">{safe_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============= 8. Streamlit chat UI =============
def main():
    st.set_page_config(page_title="Rainfall Chat Assistant", page_icon="🌧️")
    inject_css()

    st.title("🌧️ Daily Rainfall Prediction Chat Assistant")
    st.write(
        "Ask about the rainfall on a specific day.\n\n"
        "Examples:\n"
        "- `San Francisco 2019-01-15`\n"
        "- `New York March 1 2020`\n"
        "- `SF 2020/1/12`"
    )

    model, device, daily_ds, lookback = load_model_and_data()

    # Keep chat history in session_state
    if "history" not in st.session_state:
        st.session_state.history = []

    # 1) Get new user input
    user_input = st.chat_input("Type a city and date, e.g. San Francisco 2019-01-15")

    if user_input:
        # store user message
        st.session_state.history.append(("user", user_input))
        # get LLM answer (with ML context if available)
        answer = answer_question(user_input, model, device, daily_ds, lookback)
        st.session_state.history.append(("assistant", answer))

    # 2) Render entire history with bubbles
    render_history(st.session_state.history)


if __name__ == "__main__":
    main()
