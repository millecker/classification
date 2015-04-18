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

import java.util.Map;

public class Item extends FeatureVector {

  private Long m_id;
  private Integer m_actualClass = null;
  private Integer m_predictedClass = null;
  private Map<Integer, Double> m_predictedClassProbabilities = null;

  public Item(Long id, Map<Integer, Double> featureVector, Integer actualClass) {
    super(featureVector);
    this.m_id = id;
    this.m_actualClass = actualClass;
  }

  public Long getId() {
    return m_id;
  }

  public Integer getActualClass() {
    return m_actualClass;
  }

  public Integer getPredictedClass() {
    return m_predictedClass;
  }

  public void setPredictedClass(Integer predictedClass) {
    this.m_predictedClass = predictedClass;
  }

  public Map<Integer, Double> getPredictedClassProbabilities() {
    return m_predictedClassProbabilities;
  }

  public void setPredictedClassProbabilities(
      Map<Integer, Double> predictedClassProbabilities) {
    this.m_predictedClassProbabilities = predictedClassProbabilities;
  }

  @Override
  public String toString() {
    return "Item [id=" + m_id + ", actualClass=" + m_actualClass
        + ", predictedClass=" + m_predictedClass + ", featureVector: "
        + this.getFeatureVector() + "]";
  }

}
