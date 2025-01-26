from django.shortcuts import render
from django.http import JsonResponse
from main_app import displayregion

def main_view(request):
    result = displayregion.main()  # Call a function from your existing main.py

    return JsonResponse({"result": result})