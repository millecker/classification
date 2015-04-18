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

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.zip.GZIPInputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IOUtils {
  private static final int GZIP_FILE_BUFFER_SIZE = 65536;
  private static final Logger LOG = LoggerFactory.getLogger(IOUtils.class);

  public static boolean exists(String file) {
    // 1) check if file is in jar
    if (IOUtils.class.getClassLoader().getResourceAsStream(file) != null) {
      return true;
    }
    // windows File.separator is \, but getting resources only works with /
    if (IOUtils.class.getClassLoader().getResourceAsStream(
        file.replaceAll("\\\\", "/")) != null) {
      return true;
    }

    // 2) if not found in jar, check the file system
    return new File(file).exists();
  }

  public static InputStream getInputStream(String fileOrUrl) {
    return getInputStream(fileOrUrl, false);
  }

  public static InputStream getInputStream(String fileOrUrl, boolean unzip) {
    InputStream in = null;
    try {
      if (fileOrUrl.matches("https?://.*")) {
        URL u = new URL(fileOrUrl);
        URLConnection uc = u.openConnection();
        in = uc.getInputStream();
      } else {
        // 1) check if file is within jar
        in = IOUtils.class.getClassLoader().getResourceAsStream(fileOrUrl);

        // windows File.separator is \, but getting resources only works with /
        if (in == null) {
          in = IOUtils.class.getClassLoader().getResourceAsStream(
              fileOrUrl.replaceAll("\\\\", "/"));
        }

        // 2) if not found in jar, load from the file system
        if (in == null) {
          in = new FileInputStream(fileOrUrl);
        }
      }

      // unzip if necessary
      if ((unzip) && (fileOrUrl.endsWith(".gz"))) {
        in = new GZIPInputStream(in, GZIP_FILE_BUFFER_SIZE);
      }

      // buffer input stream
      in = new BufferedInputStream(in);

    } catch (FileNotFoundException e) {
      LOG.error("FileNotFoundException: " + e.getMessage());
    } catch (IOException e) {
      LOG.error("IOException: " + e.getMessage());
    }

    return in;
  }

  public static InputStream getInputStream(File file) {
    InputStream in = null;
    try {
      in = new FileInputStream(file);

      // unzip if necessary
      if (file.getName().endsWith(".gz")) {
        in = new GZIPInputStream(in, GZIP_FILE_BUFFER_SIZE);
      }

      // buffer input stream
      in = new BufferedInputStream(in);

    } catch (FileNotFoundException e) {
      LOG.error("FileNotFoundException: " + e.getMessage());
    } catch (IOException e) {
      LOG.error("IOException: " + e.getMessage());
    }

    return in;
  }

}
