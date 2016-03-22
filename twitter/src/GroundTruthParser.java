import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

/**
 * Created by apals on 22/03/16.
 */
public class GroundTruthParser {
    public static void main(String[] arg) throws IOException {

        Scanner sc = new Scanner(new File("tweets_GroundTruth.txt"));
        File file = new File("tweets_GroundTruth-parsed.txt");
        FileWriter fw = new FileWriter(file);
        sc.nextLine();

        String str;
        int j = 0;
        while (sc.hasNextLine()) {
            str = "";
            String[] line = sc.nextLine().split("\t");


            double score = Double.parseDouble(line[1])/4;
            System.out.println("---------------------------");
            System.out.println("SCORE: " + score);
            System.out.println("Tweets: " + line[2]);



            String sentiment = score > 0.2 ? "positive" : score < -0.2 ? "negative" : "neutral";
            //System.out.println("score: " + score + ", sent: " + sentiment);



            fw.write(line[2] + "\n");
            fw.write("\"" + sentiment + "\"\n");
            /*;
            ;*/
            j++;

        }
        sc.close();
        fw.close();
        System.out.println(j);
    }
}
