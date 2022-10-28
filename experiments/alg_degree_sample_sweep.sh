#!/bin/bash
# Parameter sweep of maximum graph degree and maximum sample size per node
cd ~/graphsage-sparse

# BTER
n=1
d='bter'
p='gpu'
as=('dense' 'sparse')
b=256
e=5
ad=256
de=2
ms=(20 50 100)
nss=5
ss=(5 15 30)
do=.5

echo 'BTER'
for a in "${as[@]}"; do
    echo "Algorithm:  ${a}"
    for m in "${ms[@]}"; do
        echo "Max Degree:  ${m}"
        for s in "${ss[@]}"; do
            echo "Sample Size:  ${s}"
            jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
            python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -nss ${nss} -s ${s} -do ${do} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
        done
    done
done

#LREDDIT
n=1
as=('dense' 'sparse')
d='lreddit'
p='gpu'
b=256
e=5
ad=128
de=2
ms=(32 64 128)
nss=5
ss=(5 15 30)
do=.5
pt=.01

echo 'LREDDIT'
for a in "${as[@]}"; do
    echo "Algorithm:  ${a}"
    for m in "${ms[@]}"; do
        echo "Max Degree:  ${m}"
        for s in "${ss[@]}"; do
            echo "Sample Size:  ${s}"
            jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
            python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -nss ${nss} -s ${s} -do ${do} -pt ${pt} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
        done
    done
done

#ARXIV
n=1
as=('dense' 'sparse')
d='arxiv'
p='gpu'
b=256
e=200
ad=256
de=2
ms=(20 100 400)
ss=(5 15 30)
pa=20
do=.5

echo 'ARXIV'
for a in "${as[@]}"; do
    echo "Algorithm:  ${a}"
    for m in "${ms[@]}"; do
        echo "Max Degree:  ${m}"
        for s in "${ss[@]}"; do
            echo "Sample Size:  ${s}"
            jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
            python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -s ${s} -do ${do} -pa ${pa} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
        done
    done
done

#NREDDIT
n=1
as=('dense' 'sparse')
d='nreddit'
p='gpu'
bs=256
e=200
ad=128
de=2
ms=(20 100 400)
ss=(5 15 30)
pa=20
do=.5

echo 'NREDDIT'
for a in "${as[@]}"; do
    echo "Algorithm:  ${a}"
    for m in "${ms[@]}"; do
        echo "Max Degree:  ${m}"
        for s in "${ss[@]}"; do
            echo "Sample Size:  ${s}"
            jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
            python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -s ${s} -do ${do} -pa ${pa} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
        done
    done
done
