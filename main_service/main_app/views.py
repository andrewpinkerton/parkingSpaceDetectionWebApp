from django.shortcuts import render
from django.http import JsonResponse
from django.template import loader
from django.http import HttpResponse

from main_app import displayregion

def main_view(request):
    homeTemplate = loader.get_template('index.html')

    return HttpResponse(homeTemplate.render())

def process_view(request):
    parkingLot = request.GET.get('process', '')  # Get value from query parameters
    if not parkingLot:
        return JsonResponse({"error": "No parking lot provided"}, status=400)
    homeTemplate = loader.get_template('index.html')

    result = displayregion.main(parkingLot)  # Call function correctly

    print(result)
    return render(request, 'index.html', {"result": result})
