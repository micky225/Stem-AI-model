import os
import joblib
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


MODEL_PATH = os.path.join(os.path.dirname(__file__), '../learning_path_model.pkl')
model = joblib.load(MODEL_PATH)



def validate_input(data):
    required_fields = ['Math', 'Physics', 'Chemistry', 'Biology', 'Coding', 'Engineering']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        try:
            data[field] = float(data[field])  
        except ValueError:
            raise ValueError(f"Invalid value for {field}: must be a number")
    return data



def detect_strengths_weaknesses(data, top_n=2, bottom_n=2):
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    strengths = [subject for subject, score in sorted_items[:top_n]]
    weaknesses = [subject for subject, score in sorted_items[-bottom_n:]]
    return strengths, weaknesses



def analyze_student_profile(data):
    subject_scores = {
        'Math': data['Math'],
        'Physics': data['Physics'],
        'Chemistry': data['Chemistry'],
        'Biology': data['Biology'],
        'Coding': data['Coding'],
        'Engineering': data['Engineering']
    }

    input_features = np.array([[subject_scores['Math'], subject_scores['Physics'], subject_scores['Chemistry'],
                                subject_scores['Biology'], subject_scores['Coding'], subject_scores['Engineering']]])

    predicted_interest = model.predict(input_features)[0]
    strengths, weaknesses = detect_strengths_weaknesses(subject_scores)

    return {
        "predicted_learning_path": predicted_interest,
        "strengths": strengths,
        "weaknesses": weaknesses
    }



@api_view(['POST'])
def predict_learning_path(request):
    try:
        data = validate_input(request.data)
        result = analyze_student_profile(data)
        return Response(result, status=status.HTTP_200_OK)
    except ValueError as ve:
        return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": "An error occurred during prediction."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
