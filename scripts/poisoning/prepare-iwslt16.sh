#!/bin/bash

# Retrieve global variables
source globals.sh

# Number of operations for BPE (more ops = bigger vocab)
BPE_N_OPS=30000

# Create folders
mkdir -p $IWSLT2016_DIR
mkdir -p $IWSLT2016_DIR/orig
mkdir -p $IWSLT2016_DIR/test

# Helper function to remove xml tags from the training files
function strip_xml_train () {
  sed '/^\s*</d' $1 > $2
  sed -i -e 's/^\s*//g' $2
  sed -i -e 's/\s*$//g' $2
}


# Helper function to remove xml tags from the test files
function strip_xml_test () {
  # Keep only segs
  sed '/<seg/!d' $1 > $2
  sed -i -e 's/\s*<[^>]*>\s*//g' $2
}

# Iterate over language pairs (you could add ar-en as well)
for lang_pair in "de-en" "fr-en" "cs-en"
do
  # Destination
  dir=$IWSLT2016_DIR/$lang_pair
  codes_dir=$IWSLT2016_DIR/$lang_pair/codes
  orig_dir=$IWSLT2016_DIR/orig/$lang_pair
  test_dir=$IWSLT2016_DIR/test/$lang_pair
  # Source and target language
  src=`echo $lang_pair | cut -d"-" -f1`
  trg=`echo $lang_pair | cut -d"-" -f2`

  # Download and extract
  # -------------------

  # Download train & dev
  wget -nc "https://wit3.fbk.eu/archive/2016-01//texts/$src/$trg/$lang_pair.tgz" -O "$lang_pair-train.tgz"
  tar xvzf "$lang_pair-train.tgz" -C $IWSLT2016_DIR/orig

  # Download test
  wget -nc "https://wit3.fbk.eu/archive/2016-01-test//texts/en/${src}/en-${src}.tgz"
  wget -nc "https://wit3.fbk.eu/archive/2016-01-test//texts/${src}/en/${src}-en.tgz"
  tar xvzf  "en-${src}.tgz" -C $IWSLT2016_DIR/test
  tar xvzf  "${src}-en.tgz" -C $IWSLT2016_DIR/test
  mv $IWSLT2016_DIR/test/en-${src}/IWSLT16.TED.tst2015.en-${src}.en.xml $IWSLT2016_DIR/test/$lang_pair/IWSLT16.TED.tst2015.${src}-en.en.xml
  mv $IWSLT2016_DIR/test/en-${src}/IWSLT16.TED.tst2016.en-${src}.en.xml $IWSLT2016_DIR/test/$lang_pair/IWSLT16.TED.tst2016.${src}-en.en.xml

  # Cleanup and compile train/dev/test
  # ---------------------------------

  mkdir -p $dir

  # Strip lines with xml from the training set (also strip trailing spaces)
  echo "Removing XML from the training data"
  for lang in $src $trg
  do
    if [ ! -x $dir/train.$lang ]
    then
      strip_xml_train $orig_dir/train.tags.$lang_pair.$lang $dir/train.$lang
    fi
  done
  # Strip xml tags from dev files
  echo "Removing XML from the validation data"
  for dev_xml in $orig_dir/*.xml
  do
    strip_xml_test $dev_xml ${dev_xml%.xml}
  done
  # Concatenate all dev files
  echo "Creating dev set"
  cat $orig_dir/IWSLT16.*.$src > $dir/valid.$src
  cat $orig_dir/IWSLT16.*.$trg > $dir/valid.$trg

  # Strip xml tags from test files
  echo "Removing XML from the test data"
  for test_xml in $IWSLT2016_DIR/test/$lang_pair/*.xml
  do
    strip_xml_test $test_xml ${test_xml%.xml}
  done
  # Test files = tst2015
  echo "Creating test set"
  cat $test_dir/IWSLT16.TED.tst201{5,6}.$lang_pair.$src > $dir/test.$src
  cat $test_dir/IWSLT16.TED.tst201{5,6}.$lang_pair.$trg > $dir/test.$trg

  # Preprocessing
  # -------------

  # Tokenize
  for lang in $src $trg
  do
    for split in train test valid
    do
      if [ ! -x $dir/$split.tok.$lang ]
      then
        echo "Tokenizing the $split data"
        $MOSES_SCRIPTS/tokenizer/tokenizer.perl -threads 5 -l $lang \
          < $dir/$split.$lang \
          > $dir/$split.tok.$lang
      fi
    done
  done

  # BPE
  mkdir $codes_dir
  for lang in $src $trg
  do
    # Learn BPE
    if [ ! -x $codes_dir/codes.$lang ]
    then
      echo "Learning $lang BPE model"
      $SUBWORD_NMT/learn-bpe -s $BPE_N_OPS < $dir/train.tok.$lang > $codes_dir/codes.$lang
    fi
    # Apply BPE
    for split in train test valid
    do
      if [ ! -x $dir/$split.bpe.$lang ]
      then
        echo "Applying BPE to $split.$lang"
        $SUBWORD_NMT/apply-bpe -c $codes_dir/codes.$lang \
          < $dir/$split.tok.$lang \
          > $dir/$split.bpe.$lang
      fi
    done
  done

  # Clean-up download files
  echo "Removing data archives"
  rm $lang_pair*.tgz
done

# Final cleanup
rm -r $IWSLT2016_DIR/orig
rm -r $IWSLT2016_DIR/test
