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
package at.illecker.classification.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import at.illecker.classification.commons.Dataset;
import at.illecker.classification.commons.Item;

public class FileUtils {
  private static final Logger LOG = LoggerFactory.getLogger(FileUtils.class);

  public static List<Item> readItems(String file, Dataset property,
      boolean readActualClass) {
    return readItems(IOUtils.getInputStream(file), property, readActualClass);
  }

  public static List<Item> readItems(InputStream is, Dataset dataset,
      boolean readActualClass) {
    List<Item> items = new ArrayList<Item>();
    InputStreamReader isr = null;
    BufferedReader br = null;
    if (is == null) {
      LOG.error("InputStream is null! File could not be found!");
      return null;
    }
    try {
      isr = new InputStreamReader(is, "UTF-8");
      br = new BufferedReader(isr);
      String line = "";
      long lineCounter = 0;
      while ((line = br.readLine()) != null) {
        lineCounter++;
        if ((lineCounter == 1) && (dataset.skipFirstLine())) {
          continue;
        }

        String[] values = line.split(dataset.getDelimiter());
        // Parse id
        Long id = null;
        try {
          id = Long.parseLong(values[dataset.getIdIndex()]);
        } catch (NumberFormatException e) {
          LOG.error("id \"" + values[dataset.getIdIndex()]
              + "\" could not be parsed!");
        }
        // Parse actualClass
        Integer actualClass = null;
        if (readActualClass) {
          String actualClassString = values[dataset.getActualClassIndex()];
          // Use regex
          if ((dataset.getActualClassRegex() != null)
              && (!dataset.getActualClassRegex().isEmpty())) {
            actualClassString = actualClassString.replaceAll(
                dataset.getActualClassRegex(), "");
          }
          try {
            actualClass = Integer.parseInt(actualClassString);
          } catch (NumberFormatException e) {
            LOG.warn("actualClass \"" + actualClassString
                + "\" could not be parsed!");
          }
          // Use offset
          if (dataset.getActualClassOffset() != null) {
            actualClass += dataset.getActualClassOffset();
          }
        }
        // Parse FeatureVector
        Map<Integer, Double> featureVector = new TreeMap<Integer, Double>();
        for (int i = dataset.getFeatureVectorStartIdx(); i <= dataset
            .getFeatureVectorEndIdx(); i++) {
          double value = Double.parseDouble(values[i]);
          if (value != 0) {
            featureVector.put(i, value);
          }
        }

        // Debug
        // LOG.info(line);
        // LOG.info(new Item(id, featureVector, actualClass).toString());
        items.add(new Item(id, featureVector, actualClass));
      }

    } catch (IOException e) {
      LOG.error("IOException: " + e.getMessage());
    } finally {
      if (br != null) {
        try {
          br.close();
        } catch (IOException ignore) {
        }
      }
      if (isr != null) {
        try {
          isr.close();
        } catch (IOException ignore) {
        }
      }
      if (is != null) {
        try {
          is.close();
        } catch (IOException ignore) {
        }
      }
    }
    LOG.info("Loaded total " + items.size() + " items");
    return items;
  }

  public static void writeItems(String file, Dataset dataset) {
    List<Item> items = dataset.getTestItems();
    String sep = dataset.getDelimiter();
    OutputStream os = null;
    OutputStreamWriter osw = null;
    BufferedWriter bw = null;
    try {
      os = new FileOutputStream(file);
      osw = new OutputStreamWriter(os, "UTF-8");
      bw = new BufferedWriter(osw);

      int i = 0;
      for (Item item : items) {
        // print header
        if (i == 0) {
          StringBuilder sbHeader = new StringBuilder("id");
          for (Map.Entry<Integer, Double> prob : item
              .getPredictedClassProbabilities().entrySet()) {
            int classLabel = prob.getKey()
                + (dataset.getActualClassOffset() * -1);
            sbHeader.append(sep + dataset.getActualClassRegex() + classLabel);
          }
          bw.write(sbHeader.toString());
          bw.newLine();
        }
        // print item
        StringBuilder sb = new StringBuilder();
        sb.append(item.getId());
        for (Map.Entry<Integer, Double> prob : item
            .getPredictedClassProbabilities().entrySet()) {
          sb.append(sep + prob.getValue());
        }
        bw.write(sb.toString());
        bw.newLine();
        i++;
      }

    } catch (IOException e) {
      LOG.error("IOException: " + e.getMessage());
    } finally {
      if (bw != null) {
        try {
          bw.close();
        } catch (IOException ignore) {
        }
      }
      if (osw != null) {
        try {
          osw.close();
        } catch (IOException ignore) {
        }
      }
      if (os != null) {
        try {
          os.close();
        } catch (IOException ignore) {
        }
      }
    }
  }

}
