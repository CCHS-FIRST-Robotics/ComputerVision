import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import edu.wpi.first.math.geometry.Pose2d; //Doesn't work on my machine
import edu.wpi.first.math.geometry.Rotation2d; //This too


public class Localization {
    // 0,0 is Bottom-Left Corner on Map (Blue Area, Red Barge)
    public static final HashMap<Integer, List<Double>> aprilTagIDs = new HashMap<>();


    //RADIANS! 🚨 0rad IS UP!!! 🚨
    public static double[] localize(double angle, double angleOffset, double distance, int aprilTagID){
        // angle is the angle from the camera to the tag
        // angleOffset is the robot's angle (0 if the robot is facing positive y. 
        //      If everything breaks, this variable is probably why since I didn't test it
        // distance is distance to the tag in inches (I think)
        // aprilTagID is the actual tag we're looking at so we know its position


        List<Double> data = aprilTagIDs.get(aprilTagID);

        // Java uses -pi to pi
        double finalAngle = angle + angleOffset;
        System.out.println(finalAngle);

        // Values before adding coordinates
        double[] relatives = {Math.sin(finalAngle) * distance, Math.cos(finalAngle) * distance};

        double[] output = {data.get(0) - relatives[0], data.get(1) - relatives[1]};
        Pose2d pose = new Pose2d(output[0], output[1], new Rotation2d(angleOffset));
        return output;
    }

    public static void main(String[] args) {
        /* This is all of the April Tag data (collaspe it if you don't wanna go insane) */{
            aprilTagIDs.put(1, Arrays.asList(657.37, 25.80, 58.50, 126.0, 0.0));
            aprilTagIDs.put(2, Arrays.asList(657.37, 291.20, 58.50, 234.0, 0.0));
            aprilTagIDs.put(3, Arrays.asList(455.15, 317.15, 51.25, 270.0, 0.0));
            aprilTagIDs.put(4, Arrays.asList(365.20, 241.64, 73.54, 0.0, 0.0));
            aprilTagIDs.put(5, Arrays.asList(365.20, 75.39, 73.54, 0.0, 30.0));
            aprilTagIDs.put(6, Arrays.asList(530.49, 130.17, 12.13, 300.0, 0.0));
            aprilTagIDs.put(7, Arrays.asList(546.87, 158.50, 12.13, 0.0, 0.0));
            aprilTagIDs.put(8, Arrays.asList(530.49, 186.83, 12.13, 0.0, 0.0));
            aprilTagIDs.put(9, Arrays.asList(497.77, 186.83, 12.13, 120.0, 0.0));
            aprilTagIDs.put(10, Arrays.asList(481.39, 158.50, 12.13, 180.0, 0.0));
            aprilTagIDs.put(11, Arrays.asList(497.77, 130.17, 12.13, 240.0, 0.0));
            aprilTagIDs.put(12, Arrays.asList(33.51, 25.80, 58.50, 54.0, 0.0));
            aprilTagIDs.put(13, Arrays.asList(33.51, 291.20, 58.50, 0.0, 0.0));
            aprilTagIDs.put(14, Arrays.asList(325.68, 241.64, 73.54, 180.0, 0.0));
            aprilTagIDs.put(15, Arrays.asList(325.68, 75.39, 73.54, 180.0, 30.0));
            aprilTagIDs.put(16, Arrays.asList(235.73, -0.15, 51.25, 0.0, 0.0));
            aprilTagIDs.put(17, Arrays.asList(160.39, 130.17, 12.13, 240.0, 0.0));
            aprilTagIDs.put(18, Arrays.asList(144.00, 158.50, 12.13, 180.0, 0.0));
            aprilTagIDs.put(19, Arrays.asList(160.39, 186.83, 12.13, 180.0, 0.0));
            aprilTagIDs.put(20, Arrays.asList(193.10, 186.83, 12.13, 120.0, 0.0));
            aprilTagIDs.put(21, Arrays.asList(209.49, 158.50, 12.13, 0.0, 0.0));
            aprilTagIDs.put(22, Arrays.asList(193.10, 130.17, 12.13, 300.0, 0.0));
        }

        // Play around with the localize parameters to test numbers

        double[] coords = localize(0, 0, 30, 1);
        System.out.println(coords[0] + ", " + coords[1] + "\n");
    }
}
