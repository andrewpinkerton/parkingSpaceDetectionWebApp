# Parking Space Detection Web App

The **Parking Space Detection Web App** for the project utilized the Django framework to receive and process GET requests served from the later implemented front end mobile app. This service implemented the preexisting parking space detection program. A GET request with a parameterized parking lot would be send to a “/process” endpoint that was either hosted locally or on an AWS EC2 instance. The service would pass the parking lot as a parameter to parking space detection program and would return a JSON object filled with the total spaces, occupied spaces, unoccupied indexes of the spaces, and occupied percentage. This JSON object would be utilized by the front-end client.

## Technologies Used
- **Django**
-  **AWS EC2 Instance**

## Screenshots

### GET Method Interface and JSON Response
<div style="display: flex; justify-content: space-between;">
  <img width="200" alt="JSON Response" src="https://github.com/user-attachments/assets/9fed44f9-27a9-41f5-be55-1a0ca7fb42c7">
  <img width="200" alt="Interface" src="https://github.com/user-attachments/assets/f1916707-c516-49f3-9396-034d75f75b26">
</div>
