setup -e 
export mode=None
for idx in {0}
do
python main_bc.py $idx $mode
done
