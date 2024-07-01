import cv2
import imutils
import pytesseract
import winsound

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract state code from number plate text
def extract_state_code(text):
    state_codes = {
        "AP": "Andhra Pradesh",
        "AR": "Arunachal Pradesh",
        "AS": "Assam",
        "BR": "Bihar",
        "CG": "Chhattisgarh",
        "GA": "Goa",
        "GJ": "Gujarat",
        "HR": "Haryana",
        "HP": "Himachal Pradesh",
        "JK": "Jammu and Kashmir",
        "JH": "Jharkhand",
        "KA": "Karnataka",
        "KL": "Kerala",
        "MP": "Madhya Pradesh",
        "MH": "Maharashtra",
        "MN": "Manipur",
        "ML": "Meghalaya",
        "MZ": "Mizoram",
        "NL": "Nagaland",
        "OD": "Odisha",
        "PB": "Punjab",
        "RJ": "Rajasthan",
        "SK": "Sikkim",
        "TN": "Tamil Nadu",
        "TS": "Telangana",
        "TR": "Tripura",
        "UP": "Uttar Pradesh",
        "UK": "Uttarakhand",
        "WB": "West Bengal",
        "AN": "Andaman and Nicobar Islands",
        "CH": "Chandigarh",
        "DN": "Dadra and Nagar Haveli and Daman and Diu",
        "DL": "Delhi",
        "LA": "Ladakh",
        "PY": "Puducherry"
    }
    for word in text.split():
        if word[:2] in state_codes:
            return word[:2], state_codes[word[:2]]
    return "Unknown", "Unknown"

# Load the image
image = cv2.imread('CarPictures/002.jpg')
image = imutils.resize(image, width=500)

# Display original image
cv2.imshow("Original Image", image)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray)

# Reduce noise from the image
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Smoother Image", gray)

# Find edges in the image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny edge", edged)

# Find contours based on the images
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

# Draw top 30 contours
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", image)

# Find the best possible contour of our expected number plate
name=1
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        NumberPlateCount = approx
        x, y, w, h = cv2.boundingRect(c)
        crp_img = image[y:y + h, x:x + w]
        cv2.imwrite(str(name) + '.png', crp_img)
        name += 1
        break

# Draw contour in our main image that we have identified as a number plate
cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)
cv2.imshow("Final Image", image)

# Crop only the part of the number plate
crop_img_loc = '1.png'
cv2.imshow("Cropped Image", cv2.imread(crop_img_loc))

# Convert image into text using pytesseract module
text = pytesseract.image_to_string(image, lang='eng+hin+ben+kan+mal+tam+tel')
state_code, state = extract_state_code(text)
print("Number is:", text)
print("State Code:", state_code)
print("State:", state)

# Beep sound for registered vehicle captured by camera and print
frequency = 2500
duration = 1200

registered = False

# Assume text is the number plate extracted from the image
print("message")
print(text)
if text and state_code in state_codes:
    registered = True

if registered:
    print("Not Registered")
    winsound.Beep(frequency, duration)
else:
    print(f"Car belongs to {state}")
    print("Registered")

cv2.waitKey(0)
