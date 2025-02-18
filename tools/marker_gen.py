import os

import cv2


def generate_single_marker(marker_dir, aruco_dict, marker_size, marker_id):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    fn = f"{marker_dir}/marker_{marker_id}.png"
    cv2.imwrite(fn, marker_img)
    marker_img = cv2.imread(fn)
    cv2.imshow("Marker", marker_img)
    print("Dimensions:", marker_img.shape)
    cv2.waitKey(0)


def generate_bulk_markers(marker_dir, aruco_dict, marker_size, num_markers):
    for marker_id in range(num_markers):
        fn = f"{marker_dir}/marker_{marker_id}.png"
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        cv2.imwrite(fn, marker_img)

    cv2.waitKey(0)
    print(f"Generated {num_markers} markers starting with 0")


def main():
    marker_dir = "markers"
    marker_dict = cv2.aruco.DICT_APRILTAG_36h11
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
    marker_size = int(input("Enter the marker size in pixels (1000): ") or 1000)

    user_input = input(
        "Press '1' to generate a single marker or '2' to generate n markers: "
    )
    os.makedirs(marker_dir, exist_ok=True)

    if user_input == "1":
        marker_id = int(input("Enter the marker ID: "))
        generate_single_marker(marker_dir, aruco_dict, marker_size, marker_id)
    elif user_input == "2":
        num_markers = int(input("Enter the number of markers to generate: ") or 1)
        generate_bulk_markers(marker_dir, aruco_dict, marker_size, num_markers)
    else:
        print("Error. Try again.")


if __name__ == "__main__":
    main()
