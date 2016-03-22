import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

/**
 * Created by apals on 22/03/16.
 */
public class Parser {

    public static void main(String[] arg) throws IOException {

        Scanner sc = new Scanner(new File("twitter-corpus.csv"));
        File file = new File("twitter-corpus-parsed.txt");
        FileWriter fw = new FileWriter(file);
        sc.nextLine();
        String[] line;
        String str;
        int j = 0;
        while (sc.hasNextLine()) {
            str = "";
            line = sc.nextLine().split(",");
            for (int i = 4; i < line.length; i++) {
                str += line[i];
            }
            if(str.length() == 0 || line[1].contains("irrelevant")) continue;
            try {
                System.out.println(str.substring(1, str.length() - 1));
            } catch(Exception e) {
                System.out.println("THE ERROR HAS OCCURED");
                System.out.println(e);
                System.out.println(str);
                System.exit(1);
            }
            fw.write(str.substring(1, str.length() - 1) + "\n");
            fw.write(line[1].substring(1, line[1].length() - 1) + "\n");
            j++;

        }
        sc.close();
        fw.close();
        System.out.println(j);
    }
}
