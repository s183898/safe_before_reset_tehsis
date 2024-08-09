import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of images to display
images_list = [
    ("CEO", [
        "static/images_1/a_photo_of_the_face_of_a_CEO_None_normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_CEO_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_CEO_None_normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_CEO_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_CEO_None_normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_CEO_afro-american_caucasian_female_male_both_5.png"
    ]),
    ("Chef", [
        "static/images_1/a_photo_of_the_face_of_a_chef_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_chef_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_chef_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_chef_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_chef_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_chef_afro-american_caucasian_female_male_both_5.png"
    ]),
    ("Cook", [
        "static/images_1/a_photo_of_the_face_of_a_cook_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_cook_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_cook_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_cook_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_cook_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_cook_afro-american_caucasian_female_male_both_5.png",
    ]),
    ("Therapist", [
        "static/images_1/a_photo_of_the_face_of_a_therapist_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_therapist_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_therapist_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_Therapist_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_therapist_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_therapist_afro-american_caucasian_female_male_both_5.png"
    ]),
    ("Pilot", [
        "static/images_1/a_photo_of_the_face_of_a_pilot__None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_pilot__afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_pilot__None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_pilot__afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_pilot__None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_pilot__afro-american_caucasian_female_male_both_5.png",
    ]),
    ("Taxi_driver", [
        "static/images_1/a_photo_of_the_face_of_a_taxi_driver_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_taxi_driver_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_taxi_driver_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_taxi_driver_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_taxi_driver_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_taxi_driver_afro-american_caucasian_female_male_both_5.png",
    ]),
    ("Firefighter", [
        "static/images_1/a_photo_of_the_face_of_a_firefighter_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_firefighter_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_firefighter_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_firefighter_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_firefighter_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_firefighter_afro-american_caucasian_female_male_both_5.png",
    ]),
    ("Housekeeper", [
        "static/images_1/a_photo_of_the_face_of_a_housekeeper_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_housekeeper_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_housekeeper_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_housekeeper_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_housekeeper_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_housekeeper_afro-american_caucasian_female_male_both_5.png",
    ]),
    ("Flight_attendant", [
        "static/images_1/a_photo_of_the_face_of_a_flight_attendant_None_Normal_3.png",
        "static/images_1/a_photo_of_the_face_of_a_flight_attendant_afro-american_caucasian_female_male_both_3.png",
        "static/images_1/a_photo_of_the_face_of_a_flight_attendant_None_Normal_4.png",
        "static/images_1/a_photo_of_the_face_of_a_flight_attendant_afro-american_caucasian_female_male_both_4.png",
        "static/images_1/a_photo_of_the_face_of_a_flight_attendant_None_Normal_5.png",
        "static/images_1/a_photo_of_the_face_of_a_flight_attendant_afro-american_caucasian_female_male_both_5.png",
    ]),

]

# Create a figure to hold all the subplots
fig, axs = plt.subplots(nrows=9, ncols=6, figsize=(10,14))  # A4 size in inches, landscape orientation

for row, (occupation, images) in enumerate(images_list):
    for col, img_path in enumerate(images):
        img = mpimg.imread(img_path)  # Placeholder for real image loading
        axs[row, col].imshow(img)
        axs[row, col].axis('off')  # Hide axes
        if col % 2 == 0:
            method = "Original"
        else:
            method = "Debiased"
        axs[row, col].set_title(f"{occupation} {method}", fontsize=7)

# Adjust the spacing between the plots



fig.tight_layout()
plt.savefig("example_occ.png")