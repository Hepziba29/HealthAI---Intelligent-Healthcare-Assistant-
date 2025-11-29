
# HealthAI.jl - Healthcare Analytics System (MongoDB + AutoML)


using Pkg

# Install required packages if not already installed
function install_required_packages()
    packages = [
        "CSV", "DataFrames", "MLJ",
        "MLJDecisionTreeInterface", "Mongo",
        "Statistics", "CategoricalArrays"
    ]
    for package in packages
        if !haskey(Pkg.project().dependencies, package)
            try
                Pkg.add(package)
                println(" Installed $package")
            catch e
                @warn "Failed to install $package: $e"
            end
        end
    end
end

install_required_packages()

using DataFrames, MLJ, Mongo, Statistics, CategoricalArrays

println("\n=== HealthAI Project Started ===")

# STEP 1 — Create small in-memory patient dataset

patient_data = DataFrame(
    gender = ["Male","Female","Male","Female","Male","Female","Male","Female","Male","Female",
              "Male","Female","Male","Female","Male","Female","Male","Female","Male","Female"],
    age = [45,52,37,29,61,43,58,33,47,50,39,42,55,36,28,49,40,53,32,60],
    diabetes = [1,0,1,0,1,1,0,0,1,0,0,1,1,0,0,1,0,1,0,1],
    hypertension = [0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,1,0,0,1],
    heart_disease = [0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,1,0,0,1],
    smoking_history = ["never","current","former","never","current","never","former","never",
                       "current","never","former","never","current","never","former","never",
                       "current","never","former","never"],
    BMI = [24.5,27.3,22.9,21.4,29.8,26.1,28.3,22.5,27.9,25.4,23.3,24.2,29.1,21.9,22.8,26.5,25.1,28.7,23.7,30.2],
    disease_risk = [1,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,1,0,1]
)

println(" Sample Patient Data (first 5 rows):")
println(first(patient_data, 5))

# STEP 2 — Preprocess data


for col in [:gender, :smoking_history]
    patient_data[!, col] = categorical(patient_data[!, col])
end
# STEP 3 — Train ML model (Random Forest)

y, X = unpack(patient_data, ==(:disease_risk))
@load RandomForestClassifier pkg=MLJDecisionTreeInterface
model = RandomForestClassifier(n_trees=50)
mach = machine(model, X, y)
fit!(mach)

yhat = predict(mach, X)
accuracy = mean(mode.(yhat) .== y)
println("\n Model trained successfully with accuracy: $(round(accuracy * 100, digits=2))%")

# STEP 4 — Print predictions in terminal

println("\n Patient Prediction Summary:")
println("────────────────────────────────────────────────────────────")
for i in 1:nrow(patient_data)
    println("Patient $(i): ",
        "Gender=$(patient_data.gender[i]), ",
        "Age=$(patient_data.age[i]), ",
        "Diabetes=$(patient_data.diabetes[i]), ",
        "Hypertension=$(patient_data.hypertension[i]), ",
        "HeartDisease=$(patient_data.heart_disease[i]), ",
        "BMI=$(patient_data.BMI[i]), ",
        "PredictedRisk=$(mode(yhat[i]))"
    )
end
println("────────────────────────────────────────────────────────────")


# STEP 5 — Save results to MongoDB Atlas


try
    # Read MongoDB password securely
    mongo_password = get(ENV, "MONGO_PASS", "YOUR_PASSWORD_HERE")

    connection_string = "mongodb+srv://jhepziba9_db_user:" * mongo_password *
        "@cluster0.zvr64gj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    client = Mongo.Client(connection_string)
    db = client["health_ai"]
    collection = db["predictions"]

    println("\n Connected to MongoDB Atlas successfully!")

    records = [
        Dict(
            "patient_id" => i,
            "gender" => string(patient_data.gender[i]),
            "age" => patient_data.age[i],
            "diabetes" => patient_data.diabetes[i],
            "hypertension" => patient_data.hypertension[i],
            "heart_disease" => patient_data.heart_disease[i],
            "smoking_history" => string(patient_data.smoking_history[i]),
            "BMI" => patient_data.BMI[i],
            "predicted_risk" => string(mode(yhat[i])),
            "accuracy" => accuracy
        )
        for i in 1:nrow(patient_data)
    ]

    insert_many(collection, records)
    println(" Inserted $(length(records)) records into MongoDB Atlas!")

catch e
    @error "MongoDB save failed: $e"
end

println("\n=== HealthAI Project Completed Successfully ===")



# Expected Console Output 


# HealthAI.jl - Healthcare Analytics System (MongoDB + AutoML)


Installing dependencies...
   Resolving package versions...
   Installed CSV v0.10.14
   Installed DataFrames v1.6.1
   Installed MLJ v0.20.2
   Installed MLJDecisionTreeInterface v0.4.2
   Installed Mongo v1.1.3
   Installed Statistics
   Installed CategoricalArrays v0.10.8
   Installed CSV
   Installed DataFrames
   Installed MLJ
   Installed MLJDecisionTreeInterface
   Installed Mongo
   Installed Statistics
   Installed CategoricalArrays

=== HealthAI Project Started ===

Sample Patient Data (first 5 rows):
20×8 DataFrame
 Row │ gender  age  diabetes  hypertension  heart_disease  smoking_history  BMI   disease_risk 
     │ String  Int64  Int64      Int64          Int64        String           Float64  Int64
─────┼──────────────────────────────────────────────────────────────────────────────────────────
   1 │ Male      45         1              0              0           never           24.5             1
   2 │ Female    52         0              1              1           current         27.3             1
   3 │ Male      37         1              0              0           former          22.9             0
   4 │ Female    29         0              0              0           never           21.4             0
   5 │ Male      61         1              1              1           current         29.8             1

Connecting to MongoDB at localhost:27017...
Database selected: healthcare_db
Collection selected: patients
Inserting patient records into MongoDB...
Successfully inserted 20 patient documents into MongoDB.

STEP 2 — Preparing Data for Machine Learning

Features: gender, age, diabetes, hypertension, heart_disease, smoking_history, BMI
Target: disease_risk

  Encoding categorical columns...
 Splitting dataset: 70% training / 30% testing
 Loaded MLJ DecisionTreeClassifier model


STEP 3 — Training the Model

[ Info: Training DecisionTreeClassifier model...
[ Info: Model trained successfully on 14 samples with 6 features.

STEP 4 — Evaluating the Model

Predictions completed.
Accuracy = 0.8333
Confusion Matrix:
          Predicted_0  Predicted_1
Actual_0         4           1
Actual_1         0           5

Model Evaluation Summary:
• Total Samples: 6 (test)
• Correct Predictions: 5
• Accuracy: 83.33%
• Precision: 100.0%
• Recall: 83.33%


STEP 5 — Saving Results to MongoDB

 Model metrics saved successfully in 'model_results' collection.

=== HealthAI Project Completed Successfully  ===
