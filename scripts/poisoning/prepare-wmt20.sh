src=de
tgt=en

valid_raw_dir=/home/chang/nlp/resources/mt/wmt20/valid
test_raw_dir=/home/chang/nlp/resources/mt/wmt20/test

echo "pre-processing valid data..."

for l in $src $tgt; do

    echo "--------"
    echo $l
    echo "2009-2013..."
    cat $valid_raw_dir/newstest20*.$l > $valid_raw_dir/valid_09-13.$l

    echo "2014-2019..."
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    echo $t
    grep -h '<seg id' $valid_raw_dir/newstest*-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $valid_raw_dir/valid_14-19.$l

    cat $valid_raw_dir/valid_09-13.$l $valid_raw_dir/valid_14-19.$l > $valid_raw_dir/valid.$l
done
echo "done"


echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep -h '<seg id' $test_raw_dir/newstest*-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $test_raw_dir/test.$l
done
echo "done"