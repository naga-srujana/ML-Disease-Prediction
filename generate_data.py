"""
Run this ONLY if you don't have Training.csv and Testing.csv.
If you already have them from your original project, skip this file.
Usage: python generate_data.py
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

symptoms = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
    'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
    'burning_micturition','fatigue','weight_gain','anxiety','cold_hands_and_feets',
    'mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness',
    'sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine',
    'nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes',
    'acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes',
    'malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain',
    'weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_stiffness',
    'spots_feeling','dischromic_patches','watering_from_eyes','increased_appetite',
    'polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
    'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
    'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister',
    'red_sore_around_nose','yellow_crust_ooze','body_aches'
]

diseases = {
    'Fungal infection':        ['itching','skin_rash','nodal_skin_eruptions','dischromic_patches'],
    'Allergy':                 ['continuous_sneezing','shivering','chills','watering_from_eyes'],
    'GERD':                    ['stomach_pain','acidity','ulcers_on_tongue','vomiting','cough'],
    'Chronic cholestasis':     ['itching','vomiting','yellowish_skin','nausea','loss_of_appetite','abdominal_pain'],
    'Drug Reaction':           ['itching','skin_rash','stomach_pain','burning_micturition','fatigue'],
    'Peptic ulcer disease':    ['vomiting','indigestion','loss_of_appetite','abdominal_pain','passage_of_gases'],
    'AIDS':                    ['muscle_wasting','patches_in_throat','high_fever','extra_marital_contacts'],
    'Diabetes':                ['fatigue','weight_loss','restlessness','lethargy','irregular_sugar_level','polyuria'],
    'Gastroenteritis':         ['vomiting','sunken_eyes','dehydration','diarrhoea'],
    'Bronchial Asthma':        ['fatigue','cough','high_fever','breathlessness','family_history','mucoid_sputum'],
    'Hypertension':            ['headache','chest_pain','dizziness','loss_of_balance','lack_of_concentration'],
    'Migraine':                ['acidity','indigestion','headache','blurred_and_distorted_vision','excessive_hunger'],
    'Cervical spondylosis':    ['back_pain','weakness_in_limbs','neck_stiffness','dizziness','loss_of_balance'],
    'Paralysis (brain hemorrhage)': ['vomiting','headache','weakness_in_limbs','altered_sensorium'],
    'Jaundice':                ['itching','vomiting','fatigue','weight_loss','high_fever','yellowish_skin','dark_urine','abdominal_pain'],
    'Malaria':                 ['chills','vomiting','high_fever','sweating','headache','nausea','diarrhoea','muscle_pain'],
    'Chicken pox':             ['itching','skin_rash','fatigue','lethargy','high_fever','headache','loss_of_appetite','mild_fever','swelled_lymph_nodes','malaise','red_spots_over_body'],
    'Dengue':                  ['skin_rash','chills','joint_pain','vomiting','fatigue','high_fever','headache','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','muscle_pain'],
    'Typhoid':                 ['chills','vomiting','fatigue','high_fever','headache','nausea','constipation','abdominal_pain','diarrhoea','toxic_look'],
    'Hepatitis A':             ['joint_pain','vomiting','yellowish_skin','dark_urine','nausea','loss_of_appetite','abdominal_pain','diarrhoea','mild_fever','yellowing_of_eyes','muscle_pain'],
    'Hepatitis B':             ['itching','fatigue','lethargy','yellowish_skin','dark_urine','loss_of_appetite','abdominal_pain','yellowing_of_eyes','receiving_blood_transfusion'],
    'Hepatitis C':             ['fatigue','yellowish_skin','nausea','loss_of_appetite','yellowing_of_eyes','receiving_blood_transfusion','receiving_unsterile_injections'],
    'Hepatitis D':             ['joint_pain','vomiting','fatigue','loss_of_appetite','yellowing_of_eyes','dark_urine','nausea','abdominal_pain'],
    'Hepatitis E':             ['joint_pain','vomiting','fatigue','high_fever','yellowish_skin','dark_urine','nausea','loss_of_appetite','abdominal_pain','yellowing_of_eyes','acute_liver_failure'],
    'Alcoholic hepatitis':     ['vomiting','yellowish_skin','abdominal_pain','swelling_of_stomach','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload'],
    'Tuberculosis':            ['chills','vomiting','fatigue','weight_loss','cough','high_fever','breathlessness','sweating','loss_of_appetite','mild_fever','yellowing_of_eyes','phlegm','blood_in_sputum','rusty_sputum'],
    'Common Cold':             ['continuous_sneezing','chills','fatigue','cough','high_fever','headache','swelled_lymph_nodes','malaise','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','loss_of_smell','muscle_pain'],
    'Pneumonia':               ['chills','fatigue','cough','high_fever','breathlessness','sweating','malaise','phlegm','chest_pain','fast_heart_rate','rusty_sputum'],
    'Dimorphic hemorrhoids(piles)': ['constipation','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus'],
    'Heart attack':            ['vomiting','breathlessness','sweating','chest_pain','fast_heart_rate'],
    'Varicose veins':          ['fatigue','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','prominent_veins_on_calf'],
    'Hypothyroidism':          ['fatigue','weight_gain','cold_hands_and_feets','mood_swings','lethargy','dizziness','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremities','depression','irritability','abnormal_menstruation'],
    'Hyperthyroidism':         ['fatigue','mood_swings','weight_loss','restlessness','sweating','diarrhoea','fast_heart_rate','excessive_hunger','muscle_weakness','irritability','abnormal_menstruation'],
    'Hypoglycemia':            ['vomiting','fatigue','anxiety','sweating','headache','nausea','blurred_and_distorted_vision','excessive_hunger','drying_and_tingling_lips','slurred_speech','irritability','palpitations'],
    'Osteoarthritis':          ['joint_pain','neck_stiffness','knee_pain','hip_joint_pain','swelling_joints','painful_walking'],
    'Arthritis':               ['muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','loss_of_appetite','painful_walking'],
    'Vertigo':                 ['vomiting','headache','nausea','spinning_movements','loss_of_balance','unsteadiness'],
    'Acne':                    ['skin_rash','pus_filled_pimples','blackheads','scurring'],
    'Urinary tract infection': ['burning_micturition','bladder_discomfort','foul_smell_of_urine','continuous_feel_of_urine'],
    'Psoriasis':               ['skin_rash','joint_pain','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails'],
    'Impetigo':                ['skin_rash','high_fever','blister','red_sore_around_nose','yellow_crust_ooze'],
    'Influenza':               ['chills','fatigue','cough','high_fever','headache','sweating','breathlessness','body_aches','loss_of_appetite','runny_nose','nausea'],
}

all_symptoms_set = set(symptoms)
for d, s_list in diseases.items():
    for s in s_list:
        all_symptoms_set.add(s)

all_symptoms = sorted(list(all_symptoms_set))

def make_row(disease, symp_list):
    row = {s: 0 for s in all_symptoms}
    chosen = random.sample(symp_list, min(len(symp_list), random.randint(max(1, len(symp_list)-2), len(symp_list))))
    for s in chosen:
        if s in row:
            row[s] = 1
    row['prognosis'] = disease
    return row

rows_train, rows_test = [], []
for disease, symp_list in diseases.items():
    for _ in range(40):
        rows_train.append(make_row(disease, symp_list))
    for _ in range(10):
        rows_test.append(make_row(disease, symp_list))

df_train = pd.DataFrame(rows_train)
df_test  = pd.DataFrame(rows_test)

cols = all_symptoms + ['prognosis']
df_train = df_train[cols]
df_test  = df_test[cols]

df_train.to_csv('Training.csv', index=False)
df_test.to_csv('Testing.csv', index=False)
print(f"Generated Training.csv ({len(df_train)} rows) and Testing.csv ({len(df_test)} rows)")
print(f"Diseases: {len(diseases)}, Symptoms: {len(all_symptoms)}")
