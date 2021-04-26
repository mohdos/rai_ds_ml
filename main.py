
from helperFunctions import *

# Reading Data CSV
df = pd.read_csv('./ml_task_data.csv')

# Change CompletionDate to year from date
df['CompletionDate'] = pd.to_datetime(df.CompletionDate)
df['CompletionDate'] = df['CompletionDate'].dt.strftime('%Y')

# Change SpudDate to year from date
df['SpudDate'] = pd.to_datetime(df.SpudDate)
df['SpudDate'] = df['SpudDate'].dt.strftime('%Y')

# Remove rows that has SpudDate and CompletionDate as nan
df.dropna(how='all', subset=['SpudDate', 'CompletionDate'], inplace=True)

# Filling nan Completion date, by grouping spud first, as CompletionDate is highly dependant on SpudDate
spud_date_grouped = df.groupby(['SpudDate'])
for key, item in spud_date_grouped:
    df.loc[(df['SpudDate'] == key), 'CompletionDate'] = item['CompletionDate'].fillna(item['CompletionDate'].mode()[0])


# Filling nan Basin, by grouping State first, as Basin is highly dependant on State
state_grouped = df.groupby(["State"])
for key, item in state_grouped:
    state = key
    basin_mode = item["Basin"].mode()[0]
    df.loc[(df["State"] == state), "Basin"] = item["Basin"].fillna(basin_mode)
    

# Filling nan State, Latitude and Longitude, by grouping Basin first, as they are all highly dependant on Basin
basin_grouped = df.groupby(["Basin"])
for key, item in basin_grouped:
    basin = key
    state_mode = item["State"].mode()[0]
    df.loc[(df["Basin"] == basin), "State"] = item["State"].fillna(state_mode)
    df.loc[(df["Basin"] == basin), "Latitude"] = item["Latitude"].fillna(item["Latitude"].median())
    df.loc[(df["Basin"] == basin), "Longitude"] = item["Longitude"].fillna(item["Longitude"].median())


# Filling nan BVHH and LateralLengthInMiles, by grouping Basin and CompletionDate, as BVHH and LateralLengthInMiles are dependent on Basin and CompletionDate
basin_compdate_grouped = df.groupby(["Basin", "CompletionDate"])
for key, item in basin_compdate_grouped:
    basin = key[0]
    compdate = key[1]

    bvhh_median = item["BVHH"].median()
    lat_length_in_miles_median = item['LateralLengthInMiles'].median()

    if math.isnan(bvhh_median):
        bvhh_median = df.loc[(df["Basin"] == basin)]["BVHH"].median()
    
    if math.isnan(bvhh_median):
        bvhh_median = df.loc[(df["CompletionDate"] == compdate)]["BVHH"].median()
    
    if math.isnan(bvhh_median):
        bvhh_median = df["BVHH"].median()

    if math.isnan(bvhh_median):
        bvhh_median = 1


    if math.isnan(lat_length_in_miles_median):
        lat_length_in_miles_median = df.loc[(df["Basin"] == basin)]["LateralLengthInMiles"].median()
    
    if math.isnan(lat_length_in_miles_median):
        lat_length_in_miles_median = df.loc[(df["CompletionDate"] == compdate)]["LateralLengthInMiles"].median()
    
    if math.isnan(lat_length_in_miles_median):
        lat_length_in_miles_median = df["LateralLengthInMiles"].median()

    if math.isnan(lat_length_in_miles_median):
        lat_length_in_miles_median = 0.75

    
    # Updating bvhh nan values
    df.loc[(df["Basin"] == basin) & (df["CompletionDate"] == compdate), "BVHH"] = item['BVHH'].fillna(bvhh_median)

    # Updating LateralLengthInMiles nan values
    df.loc[(df["Basin"] == basin) & (df["CompletionDate"] == compdate), "LateralLengthInMiles"] = item['LateralLengthInMiles'].fillna(lat_length_in_miles_median)
    

# Filling nan formationAlias, by grouping State first, as formationAlias is highly dependant on State
state_grouped = df.groupby(["State"])
for key, item in state_grouped:
    state = key
    
    formation_alias_mode_arr = item['formationAlias'].mode()
    
    if len(formation_alias_mode_arr) == 0: # check if mode is nan
        formation_alias_mode_arr = df["formationAlias"].mode()

    df.loc[(df['State'] == state), 'formationAlias'] = item['formationAlias'].fillna(formation_alias_mode_arr[0])


# Filling nan Operator Alias by getting the mode of all OperatorAlias column, and replacing the nan values by the mode
df['OperatorAlias'] = df['OperatorAlias'].fillna(df['OperatorAlias'].mode()[0])

# Drop rows that has a column with nan
df.dropna(inplace=True)

# change CompletionDate and SpudDate to int
df['CompletionDate'] = df['CompletionDate'].astype(int)
df['SpudDate'] = df['SpudDate'].astype(int)


# generate train and test data, by giving the required x columns (features), the y column (output), encoding type (one hot / categorization, normalized or not, percent of test size)
X_train, X_test, Y_train, Y_test = generate_train_test(df, 
                                    xcolumns=['State', 'Latitude', 'Longitude', 'OperatorAlias', 'formationAlias', 'CompletionDate', 'SpudDate'], 
                                    ycolumn='proppantPerFoot', 
                                    encoding_type=encodingTypes.one_hot_encoding, 
                                    normalized=True,
                                    test_size=0.15)

# Random forest regression
model = RandomForestRegressor(n_estimators=100, max_features=7)
model.fit(X_train, Y_train)

# predict the test data
y_pred = model.predict(X_test)
ytest_list = Y_test.to_list()

# Calculate metrics (R2 score and Root mean squared error)
r2score = r2_score(y_true=ytest_list, y_pred=y_pred)
mse = np.sqrt(mean_squared_error(ytest_list,y_pred))

print('Root Mean squared error =', mse)
print('R2 Score =', r2score)

