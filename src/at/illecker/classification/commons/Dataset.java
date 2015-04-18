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
package at.illecker.classification.commons;

import java.io.File;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import libsvm.svm_parameter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import at.illecker.classification.io.FileUtils;
import at.illecker.classification.io.IOUtils;
import at.illecker.classification.io.SerializationUtils;
import at.illecker.classification.svm.SVM;

public class Dataset implements Serializable {
  private static final long serialVersionUID = -3515203143574868606L;
  private static final Logger LOG = LoggerFactory.getLogger(Dataset.class);

  private String m_datasetPath;
  private String m_trainDataFile;
  private String m_testDataFile;

  private boolean m_skipFirstLine;
  private String m_delimiter;
  private int m_idIndex;
  private int m_actualClassIndex;
  private String m_actualClassRegex;
  private Integer m_actualClassOffset;
  private int m_featureVectorStartIdx;
  private int m_featureVectorEndIdx;

  private List<Item> m_trainItems;
  private List<Item> m_testItems;

  private svm_parameter m_svmParam;

  public Dataset(String datasetPath, String trainDataFile, String testDataFile,
      boolean skipFirstLine, String delimiter, int idIndex,
      int actualClassIndex, String actualClassRegex, Integer actualClassOffset,
      int featureVectorStartIdx, int featureVectorEndIdx, svm_parameter svmParam) {
    this.m_datasetPath = datasetPath;
    this.m_trainDataFile = trainDataFile;
    this.m_testDataFile = testDataFile;

    this.m_skipFirstLine = skipFirstLine;
    this.m_delimiter = delimiter;
    this.m_idIndex = idIndex;
    this.m_actualClassIndex = actualClassIndex;
    this.m_actualClassRegex = actualClassRegex;
    this.m_actualClassOffset = actualClassOffset;
    this.m_featureVectorStartIdx = featureVectorStartIdx;
    this.m_featureVectorEndIdx = featureVectorEndIdx;

    this.m_svmParam = svmParam;
  }

  public String getDatasetPath() {
    return m_datasetPath;
  }

  public String getTrainDataFile() {
    return (m_trainDataFile != null) ? m_datasetPath + File.separator
        + m_trainDataFile : null;
  }

  public String getTrainDataSerializationFile() {
    return (m_trainDataFile != null) ? m_datasetPath + File.separator
        + m_trainDataFile + Configuration.SERIAL_EXTENSION : null;
  }

  public String getTestDataFile() {
    return (m_testDataFile != null) ? m_datasetPath + File.separator
        + m_testDataFile : null;
  }

  public String getTestDataSerializationFile() {
    return (m_testDataFile != null) ? m_datasetPath + File.separator
        + m_testDataFile + Configuration.SERIAL_EXTENSION : null;
  }

  public boolean skipFirstLine() {
    return m_skipFirstLine;
  }

  public String getDelimiter() {
    return m_delimiter;
  }

  public int getIdIndex() {
    return m_idIndex;
  }

  public int getActualClassIndex() {
    return m_actualClassIndex;
  }

  public String getActualClassRegex() {
    return m_actualClassRegex;
  }

  public Integer getActualClassOffset() {
    return m_actualClassOffset;
  }

  public int getFeatureVectorStartIdx() {
    return m_featureVectorStartIdx;
  }

  public int getFeatureVectorEndIdx() {
    return m_featureVectorEndIdx;
  }

  public svm_parameter getSVMParam() {
    return m_svmParam;
  }

  public List<Item> getTrainItems() {
    if ((m_trainItems == null) && (getTrainDataFile() != null)) {
      // Try deserialization of file
      String serializationFile = getTrainDataSerializationFile();
      if (IOUtils.exists(serializationFile)) {
        LOG.info("Deserialize TrainItems from: " + serializationFile);
        m_trainItems = SerializationUtils.deserialize(serializationFile);
      } else {
        LOG.info("Read TrainItems from: " + getTrainDataFile());
        m_trainItems = FileUtils.readItems(getTrainDataFile(), this, true);
      }
    }
    return m_trainItems;
  }

  public List<Item> getTestItems() {
    if ((m_testItems == null) && (getTestDataFile() != null)) {
      // Try deserialization of file
      String serializationFile = getTestDataSerializationFile();
      if (IOUtils.exists(serializationFile)) {
        LOG.info("Deserialize TestItems from: " + serializationFile);
        m_testItems = SerializationUtils.deserialize(serializationFile);
      } else {
        LOG.info("Read TestItems from: " + getTestDataFile());
        m_testItems = FileUtils.readItems(getTestDataFile(), this, false);
      }
    }
    return m_testItems;
  }

  public void setTestItems(List<Item> testItems) {
    m_testItems = testItems;
  }

  public void printDatasetStats() {
    LOG.info("Dataset: " + getDatasetPath());

    LOG.info("Train Items: " + getTrainDataFile());
    printTweetStats(getTrainItems());

    LOG.info("Test Items: " + getTestDataFile());
    // Load test items
    getTestItems();
  }

  public static void printTweetStats(List<Item> items) {
    if (items != null) {
      Map<Integer, Integer> counts = new TreeMap<Integer, Integer>();
      for (Item item : items) {
        int key = item.getActualClass();
        Integer count = counts.get(key);
        counts.put(key, ((count != null) ? count + 1 : 1));
      }

      int total = 0;
      int max = 0;
      for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
        LOG.info("Class: \t" + entry.getKey() + "\t" + entry.getValue());
        total += entry.getValue();
        if (entry.getValue() > max) {
          max = entry.getValue();
        }
      }
      LOG.info("Total: " + total);

      LOG.info("Optimal Class Weights: ");
      for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
        LOG.info("Class: \t" + entry.getKey() + "\t"
            + (max / (double) entry.getValue()));
      }
    }
  }

  @Override
  public String toString() {
    return "Dataset [datasetPath=" + m_datasetPath + ", trainDataFile="
        + m_trainDataFile + ", testDataFile=" + m_testDataFile + ", delimiter="
        + m_delimiter + ", idIndex=" + m_idIndex + ", actualClassIndex="
        + m_actualClassIndex + ", actualClassRegex=" + m_actualClassRegex
        + ", featureVectorStartIdx=" + m_featureVectorStartIdx
        + ", featureVectorEndIdx=" + m_featureVectorEndIdx + "]";
  }

  @SuppressWarnings({ "rawtypes", "unchecked" })
  public static Dataset readFromYaml(Map dataset) {
    svm_parameter svmParam = SVM.getDefaultParameter();

    if (dataset.get("svm.kernel") != null) {
      svmParam.kernel_type = (Integer) dataset.get("svm.kernel");
    }

    if (dataset.get("svm.c") != null) {
      svmParam.C = (Double) dataset.get("svm.c");
    }

    if (dataset.get("svm.gamma") != null) {
      svmParam.gamma = (Double) dataset.get("svm.gamma");
    }

    if (dataset.get("svm.class.weights") != null) {
      Map<Integer, Double> classWeights = (Map<Integer, Double>) dataset
          .get("svm.class.weights");

      svmParam.nr_weight = classWeights.size();
      svmParam.weight_label = new int[svmParam.nr_weight];
      svmParam.weight = new double[svmParam.nr_weight];

      for (Map.Entry<Integer, Double> entry : classWeights.entrySet()) {
        svmParam.weight_label[entry.getKey()] = entry.getKey();
        svmParam.weight[entry.getKey()] = entry.getValue();
      }
    }

    return new Dataset((String) dataset.get("path"),
        (String) dataset.get("train.file"), (String) dataset.get("test.file"),
        (Boolean) dataset.get("skipFirstLine"),
        (String) dataset.get("delimiter"), (Integer) dataset.get("id.index"),
        (Integer) dataset.get("actualClass.index"),
        (String) dataset.get("actualClass.regex"),
        (Integer) dataset.get("actualClass.offset"),
        (Integer) dataset.get("featureVectorStart.index"),
        (Integer) dataset.get("featureVectorEnd.index"), svmParam);
  }

  public static void main(String[] args) {
    List<Dataset> datasets = Configuration.getDataSets();
    for (Dataset dataset : datasets) {
      dataset.printDatasetStats();
    }
  }

}
