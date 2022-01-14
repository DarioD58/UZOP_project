import pandas as pd


def game_clock_to_seconds(game_clock):
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(game_clock.split(":"))))


def total_game_time(period, game_clock):
    if period <= 4:  # first four periods last for 12 minutes each
        total = 12 * 60 * (period - 1)
    else:  # overtime periods last for 5 minutes
        total = 60 * (12 * 4 + 5 * (period - 5))

    total += game_clock_to_seconds(game_clock)

    return total


def extract_from_matchup(matchup):
    from datetime import datetime
    # split
    date_str, teams_str = matchup.split(" - ")

    date_str = date_str.strip()
    # make month lowercase except for the first letter
    date_str = date_str[0] + date_str[1:3].lower() + date_str[3:]

    # create date from specified string format
    date = datetime.strptime(date_str, "%b %d, %Y")

    # create matchup
    if "@" in teams_str:
        return date.year, date.month, date.day, *teams_str.split(" @ ")
    elif "vs" in teams_str:
        return date.year, date.month, date.day, *teams_str.split(" vs. ")
    else:
        raise RuntimeError(f"Unknown team format: {teams_str}")


def squash_categorical_columns(df, columns):
    for column in columns:
        print(f"Squashing column: {column}")
        column_categories = df.loc[:, column].astype("category")
        df.loc[:, [column]] = column_categories.cat.codes
    return df

categorical_features = [
    "CLOSEST_DEFENDER_PLAYER_ID", 
    "GAME_ID", 
    "LOCATION",
    "GAME_EVENT_ID",
    "PLAYER_ID", 
    "TEAM_ID",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "TEAM_1",
    "TEAM_2",
    "YEAR",
    "MONTH",
    "DAY",
]

low_card_features = [
    "LOCATION", 
    "SHOT_TYPE", 
    "SHOT_ZONE_BASIC", 
    "SHOT_ZONE_AREA"
]

def one_hot_encode(df, columns):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop="first")

    for column in columns:
        encoder.fit(df.loc[:, [column]])
        one_hot_result = encoder.transform(df.loc[:, [column]]).toarray()
        one_hot_df = pd.DataFrame(one_hot_result, columns=encoder.get_feature_names_out())
        df = pd.concat([df, one_hot_df], axis=1).drop(columns=[column], axis=1)

    return df


def preprocessing(df):
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(columns=[
        "SHOT_DISTANCE",

        "MINUTES_REMAINING",
        "SECONDS_REMAINING",

        "GRID_TYPE",
        "SHOT_ATTEMPTED_FLAG",

        "CLOSEST_DEFENDER", 
        "GAME_ID.1",
        "PERIOD.1",
        "PLAYER_NAME",
        "PLAYER_ID.1",
        "PTS",
        "SHOT_MADE_FLAG",
        "EVENT_TYPE",
        "FGM",

        "SHOT_ZONE_RANGE",
        "DRIBBLES",
        "W",
        "PTS_TYPE"
        ], inplace=True)

    df.loc[:, "TOTAL_GAME_TIME"] = df.loc[:, ["PERIOD", "GAME_CLOCK"]]\
        .apply(lambda s: total_game_time(s["PERIOD"], s["GAME_CLOCK"]), axis=1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.drop(columns=["GAME_CLOCK", "PERIOD"], inplace=True)

    df.loc[:, ["YEAR", "MONTH", "DAY", "TEAM_1", "TEAM_2"]] = df.MATCHUP.apply(
        lambda row: pd.Series(
            data=extract_from_matchup(row),
            index=["YEAR", "MONTH", "DAY", "TEAM_1", "TEAM_2"]
        ))
    df.drop(columns=["MATCHUP"], inplace=True)


    for column in categorical_features:
        column_cats = df.loc[:, column].astype("category")
        df.loc[:, [column]] = column_cats.cat.codes

    df = one_hot_encode(df, low_card_features)

    return df


def prepare_input(df):
    from sklearn.preprocessing import LabelEncoder
    X = df.drop(columns=["SHOT_RESULT"]).to_numpy()
    y = LabelEncoder().fit_transform(df.loc[:, "SHOT_RESULT"])

    return X, y

def drop_seasonal_features(df):
    df.drop(columns=[
        
    ], inplace=True)