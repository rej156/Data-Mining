package classifier;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.trees.ft.FTNode;
import weka.classifiers.trees.ft.FTtree;
import weka.core.*;
import weka.core.converters.CSVLoader;

public class Classifier {

    public static weka.classifiers.Classifier rf;
    public static Evaluation eval;
    public static Instances train;
    public static Instances test;
    public static String train_file = "C:/Users/Kashif/Desktop/Computer Science/Year 3/ECS607 - Data Mining/cw/d/train_title2.csv";
    public static String test_file = "C:/Users/Kashif/Desktop/Computer Science/Year 3/ECS607 - Data Mining/cw/d/test_title2.csv";
    public static String out_file = "C:/Users/Kashif/Desktop/Computer Science/Year 3/ECS607 - Data Mining/cw/d/out.csv";

    public static CSVLoader loader;

    public static void main(String[] args) throws Exception {
        load();
        classify();
        evaluate();
        write();

        System.out.println(rf);
        System.out.println(eval.toSummaryString("Results \n", false));

    }

    public static void classify() throws Exception {
        //rf = new M5P(); UNACCEPTABLE
        
        
        //rf = new RandomForest();//77.7512
        //rf = new J48(); //77.512
        //rf = new RandomTree();
        //rf = new DecisionStump(); //76.55
        //rf = new LMT(); //76.79
        //rf = new HoeffdingTree(); //76.555
        //rf = new REPTree(); //77.7512
        //rf = new ADTree(); // 77.512
        //rf = new NBTree();//78.46
        //rf = new DecisionStump(); //75.55
        //rf = new FT(); //76.555
        //rf = new LADTree(); //77.27
        rf = new SimpleCart(); //78.489
        //rf = new BFTree();
        
        rf.buildClassifier(train);
    }

    public static void evaluate() throws Exception {
        eval = new Evaluation(test);
        eval.evaluateModel(rf, test);
    }

    public static void load() {
        // load data
        try {
            loader = new CSVLoader();
            loader.setFile(new File(train_file));
            train = loader.getDataSet();

            loader = new CSVLoader();
            loader.setFile(new File(test_file));
            test = loader.getDataSet();

            train.setClassIndex(0);
            test.setClassIndex(0);

        } catch (IOException e) {
            System.out.println("I/O ERROR");
        }
    }

    public static void write() throws Exception {
        CSVReader r = new CSVReader(new FileReader(test_file));
        r.readNext();

        CSVWriter c = new CSVWriter(new FileWriter(out_file));
        String[] out = "PassengerId,Survived".split(",");
        c.writeNext(out);
        int id = 892;
        Instances copy = new Instances(test);

        for (int x = 0; x < copy.numInstances(); x++) {
            double d = rf.classifyInstance(copy.instance(x));
            out = ((id++) + "," + d).split(",");
            c.writeNext(out);
        }

        r.close();
        c.close();

    }

}
