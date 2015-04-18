/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package at.illecker.classification.svm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import at.illecker.classification.commons.Configuration;
import at.illecker.classification.commons.Dataset;
import at.illecker.classification.commons.Item;
import at.illecker.classification.commons.Pair;
import at.illecker.classification.io.FileUtils;
import at.illecker.classification.io.SerializationUtils;

public class SVM {
  public static final String SVM_PROBLEM_FILE = "svm_problem.txt";
  public static final String SVM_MODEL_FILE_SER = "svm_model.ser";
  private static final Logger LOG = LoggerFactory.getLogger(SVM.class);

  public static svm_parameter getDefaultParameter() {
    svm_parameter param = new svm_parameter();
    // type of SVM
    param.svm_type = svm_parameter.C_SVC; // default

    // type of kernel function
    param.kernel_type = svm_parameter.RBF; // default

    // degree in kernel function (default 3)
    param.degree = 3;

    // gamma in kernel function (default 1/num_features)
    // gamma = 2^−15, 2^−13, ..., 2^3
    param.gamma = Double.MIN_VALUE;

    // parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    // C = 2^−5, 2^−3, ..., 2^15
    param.C = 1; // cost of constraints violation default 1

    // coef0 in kernel function (default 0)
    param.coef0 = 0;

    // parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    param.nu = 0.5;

    // epsilon in loss function of epsilon-SVR (default 0.1)
    param.p = 0.1;

    // tolerance of termination criterion (default 0.001)
    param.eps = 0.001;

    // whether to use the shrinking heuristics, 0 or 1
    // (default 1)
    param.shrinking = 1;

    // whether to train a SVC or SVR model for probability estimates, 0 or 1
    // (default 0)
    // 1 means model with probability information is obtained
    param.probability = 1;

    // parameter C of class i to weight*C, for C-SVC (default 1)
    param.nr_weight = 0;
    param.weight_label = new int[0];
    param.weight = new double[0];

    // cache memory size in MB (default 100)
    param.cache_size = 2000;

    return param;
  }

  public static svm_problem generateProblem(List<Item> items) {
    int dataCount = items.size();

    svm_problem svmProb = new svm_problem();
    svmProb.y = new double[dataCount];
    svmProb.l = dataCount;
    svmProb.x = new svm_node[dataCount][];

    int i = 0;
    for (Item item : items) {
      Map<Integer, Double> featureVector = item.getFeatureVector();

      // set feature nodes
      svmProb.x[i] = new svm_node[featureVector.size()];
      int j = 0;
      for (Map.Entry<Integer, Double> feature : featureVector.entrySet()) {
        svm_node node = new svm_node();
        node.index = feature.getKey();
        node.value = feature.getValue();
        svmProb.x[i][j] = node;
        j++;
      }

      // set class / label
      svmProb.y[i] = item.getActualClass();
      i++;
    }

    return svmProb;
  }

  public static void saveProblem(svm_problem svmProb, String file) {
    // save problem in libSVM format
    // <label> <index1>:<value1> <index2>:<value2> ...
    try {
      BufferedWriter br = new BufferedWriter(new FileWriter(file));
      for (int i = 0; i < svmProb.l; i++) {
        // <label>
        br.write(Double.toString(svmProb.y[i]));
        for (int j = 0; j < svmProb.x[i].length; j++) {
          if (svmProb.x[i][j].value != 0) {
            // <index>:<value>
            br.write(" " + svmProb.x[i][j].index + ":" + svmProb.x[i][j].value);
          }
        }
        br.newLine();
        br.flush();
      }
      br.close();
      LOG.info("saved svm_problem in " + file);
    } catch (IOException e) {
      LOG.error("IOException: " + e.getMessage());
    }
  }

  public static svm_model train(svm_problem svmProb, svm_parameter svmParam) {
    // set gamma to default 1/num_features if not specified
    if (svmParam.gamma == Double.MIN_VALUE) {
      svmParam.gamma = 1 / (double) svmProb.l;
    }

    String paramCheck = svm.svm_check_parameter(svmProb, svmParam);
    if (paramCheck != null) {
      LOG.error("svm_check_parameter: " + paramCheck);
    }

    return svm.svm_train(svmProb, svmParam);
  }

  public static double crossValidate(svm_problem svmProb,
      svm_parameter svmParam, int nFold) {
    return crossValidate(svmProb, svmParam, nFold, false);
  }

  public static double crossValidate(svm_problem svmProb,
      svm_parameter svmParam, int nFold, boolean printStats) {

    // set gamma to default 1/num_features if not specified
    if (svmParam.gamma == Double.MIN_VALUE) {
      svmParam.gamma = 1 / (double) svmProb.l;
    }

    double[] target = new double[svmProb.l];
    svm.svm_cross_validation(svmProb, svmParam, nFold, target);

    double correctCounter = 0;
    for (int i = 0; i < svmProb.l; i++) {
      if (target[i] == svmProb.y[i]) {
        correctCounter++;
      }
    }

    double accuracy = correctCounter / (double) svmProb.l;
    LOG.info("Cross Validation Accuracy: " + (100.0 * accuracy));

    if (printStats) {
      printStats(getConfusionMatrix(svmProb.y, target));
    }

    return accuracy;
  }

  public static void coarseGrainedParamterSearch(svm_problem svmProb,
      svm_parameter svmParam) {
    // coarse grained paramter search
    int maxC = 11;
    double[] c = new double[maxC];
    // C = 2^−5, 2^−3, ..., 2^15
    for (int i = 0; i < maxC; i++) {
      c[i] = Math.pow(2, -5 + (i * 2));
    }
    int maxGamma = 10;
    double[] gamma = new double[maxGamma];
    // gamma = 2^−15, 2^−13, ..., 2^3
    for (int j = 0; j < maxGamma; j++) {
      gamma[j] = Math.pow(2, -15 + (j * 2));
    }

    paramterSearch(svmProb, svmParam, c, gamma);
  }

  private static class FindParameterCallable implements Callable<double[]> {
    private svm_problem m_svmProb;
    private svm_parameter m_svmParam;
    private long m_i;
    private long m_j;

    public FindParameterCallable(svm_problem svmProb, svm_parameter svmParam,
        long i, long j) {
      m_svmProb = svmProb;
      m_svmParam = svmParam;
      m_i = i;
      m_j = j;
    }

    @Override
    public double[] call() throws Exception {
      long startTime = System.currentTimeMillis();
      double accuracy = crossValidate(m_svmProb, m_svmParam, 10);
      long estimatedTime = System.currentTimeMillis() - startTime;
      return new double[] { m_i, m_j, accuracy, m_svmParam.C, m_svmParam.gamma,
          estimatedTime };
    }
  }

  public static void paramterSearch(svm_problem svmProb,
      svm_parameter svmParam, double[] c, double[] gamma) {
    int cores = Runtime.getRuntime().availableProcessors();
    ExecutorService executorService = Executors.newFixedThreadPool(cores);
    Set<Callable<double[]>> callables = new HashSet<Callable<double[]>>();

    for (int i = 0; i < c.length; i++) {
      for (int j = 0; j < gamma.length; j++) {
        svm_parameter param = (svm_parameter) svmParam.clone();
        param.C = c[i];
        param.gamma = gamma[j];
        callables.add(new FindParameterCallable(svmProb, param, i, j));
      }
    }

    try {
      long startTime = System.currentTimeMillis();
      List<Future<double[]>> futures = executorService.invokeAll(callables);
      for (Future<double[]> future : futures) {
        double[] result = future.get();
        LOG.info("findParamters[" + result[0] + "," + result[1] + "] C="
            + result[3] + " gamma=" + result[4] + " accuracy: " + result[2]
            + " time: " + result[5] + " ms");
      }
      long estimatedTime = System.currentTimeMillis() - startTime;
      LOG.info("findParamters total execution time: " + estimatedTime
          + " ms - " + (estimatedTime / 1000) + " sec");

      // output CSV file
      LOG.info("CSV file of paramterSearch with C=" + Arrays.toString(c)
          + " gamma=" + Arrays.toString(gamma));
      LOG.info("i;j;C;gamma;accuracy;time_ms");
      for (Future<double[]> future : futures) {
        double[] result = future.get();
        LOG.info(result[0] + ";" + result[1] + ";" + result[3] + ";"
            + result[4] + ";" + result[2] + ";" + result[5]);
      }

    } catch (InterruptedException e) {
      LOG.error("InterruptedException: " + e.getMessage());
    } catch (ExecutionException e) {
      LOG.error("ExecutionException: " + e.getMessage());
    }

    executorService.shutdown();
  }

  public static svm_node[] getFeatureNodes(Map<Integer, Double> featureVector) {
    svm_node[] nodes = new svm_node[featureVector.size()];
    int i = 0;
    for (Map.Entry<Integer, Double> feature : featureVector.entrySet()) {
      svm_node node = new svm_node();
      node.index = feature.getKey();
      node.value = feature.getValue();
      nodes[i] = node;
      i++;
    }
    return nodes;
  }

  public static double evaluate(Map<Integer, Double> featureVector,
      svm_model svmModel) {

    svm_node[] nodes = getFeatureNodes(featureVector);
    return svm.svm_predict(svmModel, nodes);
  }

  public static Pair<Double, Map<Integer, Double>> evaluate(
      Map<Integer, Double> featureVector, svm_model svmModel, int totalClasses) {

    int[] labels = new int[totalClasses];
    svm.svm_get_labels(svmModel, labels);

    svm_node[] nodes = getFeatureNodes(featureVector);

    double[] probEstimates = new double[totalClasses];
    double predictedClassProb = svm.svm_predict_probability(svmModel, nodes,
        probEstimates);

    Map<Integer, Double> predictedClassProbabilites = new TreeMap<Integer, Double>();
    for (int i = 0; i < totalClasses; i++) {
      predictedClassProbabilites.put(labels[i], probEstimates[i]);
    }

    return new Pair<Double, Map<Integer, Double>>(predictedClassProb,
        predictedClassProbabilites);
  }

  public static int[][] getConfusionMatrix(double[] actualClass,
      double[] predictedClass) {
    if (actualClass.length != predictedClass.length) {
      return null;
    }

    // find the total number of classes
    int maxClassNum = 0;
    for (int i = 0; i < actualClass.length; i++) {
      if (actualClass[i] > maxClassNum)
        maxClassNum = (int) actualClass[i];
    }
    // add 1 because of class zero
    maxClassNum++;

    // create confusion matrix
    // rows represent the instances in an actual class
    // cols represent the instances in a predicted class
    int[][] confusionMatrix = new int[maxClassNum][maxClassNum];
    for (int i = 0; i < actualClass.length; i++) {
      confusionMatrix[(int) actualClass[i]][(int) predictedClass[i]]++;
    }
    return confusionMatrix;
  }

  public static void printStats(int[][] confusionMatrix) {
    int totalClasses = confusionMatrix.length;
    int total = 0;
    int totalCorrect = 0;

    int[] rowSum = new int[totalClasses];
    int[] colSum = new int[totalClasses];
    for (int i = 0; i < totalClasses; i++) {
      for (int j = 0; j < totalClasses; j++) {
        total += confusionMatrix[i][j];
        rowSum[i] += confusionMatrix[i][j];
        colSum[i] += confusionMatrix[j][i];
      }
      totalCorrect += confusionMatrix[i][i];
    }

    LOG.info("Confusion Matrix:");
    // print header
    StringBuffer sb = new StringBuffer();
    sb.append("\t\t");
    for (int i = 0; i < totalClasses; i++) {
      sb.append("\t").append(i);
    }
    sb.append("\t").append("total");
    LOG.info(sb.toString());
    // print matrix
    for (int i = 0; i < totalClasses; i++) {
      int[] predictedClasses = confusionMatrix[i];
      sb = new StringBuffer();
      sb.append("Class:\t" + i);
      for (int j = 0; j < predictedClasses.length; j++) {
        sb.append("\t").append(predictedClasses[j]);
      }
      sb.append("\t" + rowSum[i]);
      LOG.info(sb.toString());
    }
    sb = new StringBuffer();
    sb.append("total").append("\t");
    for (int i = 0; i < totalClasses; i++) {
      sb.append("\t").append(colSum[i]);
    }
    LOG.info(sb.toString() + "\n");

    LOG.info("Total: " + total);
    LOG.info("Correct: " + totalCorrect);
    LOG.info("Accuracy: " + (totalCorrect / (double) total));

    LOG.info("Scores per class:");
    double[] FScores = new double[totalClasses];
    for (int i = 0; i < totalClasses; i++) {
      int correctHitsPerClass = confusionMatrix[i][i];

      double precision = correctHitsPerClass / (double) colSum[i];
      double recall = correctHitsPerClass / (double) rowSum[i];
      FScores[i] = 2 * ((precision * recall) / (precision + recall));

      LOG.info("Class: " + i + " Precision: " + precision + " Recall: "
          + recall + " F-Score: " + FScores[i]);
    }

    // FScoreWeighted is a weighted average of the classes' f-scores, weighted
    // by the proportion of how many elements are in each class.
    double FScoreWeighted = 0;
    for (int i = 0; i < totalClasses; i++) {
      FScoreWeighted += FScores[i] * colSum[i];
    }
    FScoreWeighted /= total;
    LOG.info("F-Score weighted: " + FScoreWeighted);

    // F-Score average of positive and negative
    double FScoreAveragePosNeg = (FScores[0] + FScores[1]) / 2;
    LOG.info("F-Score average(pos,neg): " + FScoreAveragePosNeg);

    // Macro-average: Average precision, recall, or F1 over the classes of
    // interest.

    // Micro-average: Sum corresponding cells to create a 2 x 2 confusion
    // matrix, and calculate precision in terms of the new matrix.
    // (In this set-up, precision, recall, and F1 are all the same.)
  }

  public static void svm(Dataset dataset, int totalClasses,
      int nFoldCrossValidation, boolean parameterSearch,
      boolean useSerialization) {

    List<Item> trainItems = dataset.getTrainItems();
    List<Item> testItems = dataset.getTestItems();
    svm_parameter svmParam = dataset.getSVMParam();

    // Optional parameter search of C and gamma
    if (parameterSearch) {
      LOG.info("Generate SVM problem...");
      svm_problem svmProb = generateProblem(trainItems);

      // 1) coarse grained paramter search
      // coarseGrainedParamterSearch(svmProb, svmParam);

      // 2) fine grained paramter search
      // C = 2^6, ..., 2^12
      double[] c = new double[7];
      for (int i = 0; i < 7; i++) {
        c[i] = Math.pow(2, 6 + i);
      }
      // gamma = 2^−14, 2^−14, ..., 2^-8
      double[] gamma = new double[7];
      for (int j = 0; j < 7; j++) {
        gamma[j] = Math.pow(2, -14 + j);
      }

      LOG.info("SVM paramterSearch...");
      LOG.info("Kernel: " + svmParam.kernel_type);
      paramterSearch(svmProb, svmParam, c, gamma);

    } else {

      svm_model svmModel = null;
      LOG.info("Try loading SVM model...");
      // deserialize svmModel
      if (useSerialization) {
        svmModel = SerializationUtils.deserialize(dataset.getDatasetPath()
            + File.separator + SVM_MODEL_FILE_SER);
      }

      if (svmModel == null) {
        LOG.info("Generate SVM problem...");
        svm_problem svmProb = generateProblem(trainItems);

        // save svm problem in libSVM format
        saveProblem(svmProb, dataset.getDatasetPath() + File.separator
            + SVM_PROBLEM_FILE);

        // train model
        LOG.info("Train SVM model...");
        long startTime = System.currentTimeMillis();
        svmModel = train(svmProb, svmParam);
        LOG.info("Train SVM model finished after "
            + (System.currentTimeMillis() - startTime) + " ms");

        // serialize svm model
        if (useSerialization) {
          SerializationUtils.serialize(svmModel, dataset.getDatasetPath()
              + File.separator + SVM_MODEL_FILE_SER);
        }

        // run n-fold cross validation
        if (nFoldCrossValidation > 1) {
          LOG.info("Run n-fold cross validation...");
          startTime = System.currentTimeMillis();
          double accuracy = crossValidate(svmProb, svmParam,
              nFoldCrossValidation, true);
          LOG.info("Cross Validation finished after "
              + (System.currentTimeMillis() - startTime) + " ms");
          LOG.info("Cross Validation Accurancy: " + accuracy);
        }
      }

      // evaluate test items
      long countMatches = 0;
      int[][] confusionMatrix = new int[totalClasses][totalClasses];
      LOG.info("Evaluate test items...");

      long startTime = System.currentTimeMillis();
      for (Item testItem : testItems) {
        Map<Integer, Double> featureVector = testItem.getFeatureVector();

        Pair<Double, Map<Integer, Double>> result = evaluate(featureVector,
            svmModel, totalClasses);

        int predictedClass = result.getKey().intValue();

        testItem.setPredictedClass(predictedClass);
        testItem.setPredictedClassProbabilities(result.getValue());

        if (testItem.getActualClass() != null) {
          int actualClass = testItem.getActualClass();
          if (predictedClass == actualClass) {
            countMatches++;
          }
          confusionMatrix[actualClass][(int) predictedClass]++;
        }
      }

      // update test items in dataset
      dataset.setTestItems(testItems);

      LOG.info("Evaluate finished after "
          + (System.currentTimeMillis() - startTime) + " ms");
      LOG.info("Total test items: " + testItems.size());
      if (countMatches > 0) {
        LOG.info("Matches: " + countMatches);
        double accuracy = (double) countMatches / (double) testItems.size();
        LOG.info("Accuracy: " + accuracy);
        printStats(confusionMatrix);
      }

      svm.EXEC_SERV.shutdown();
    }
  }

  public static void main(String[] args) {
    boolean parameterSearch = false;
    boolean useSerialization = true;
    int nFoldCrossValidation = 1;

    // Get first dataset
    Dataset dataset = Configuration.getDataSets().get(0);
    dataset.printDatasetStats();

    SVM.svm(dataset, 9, nFoldCrossValidation, parameterSearch, useSerialization);

    FileUtils.writeItems(dataset.getDatasetPath() + File.separator
        + "submission.csv", dataset);
  }

}
