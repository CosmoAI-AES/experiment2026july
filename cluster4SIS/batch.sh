#!/bin/sh

D=.

if test x$1 = ximages
then
  mkdir -p $D/images
  log=$D/datagen.log 
  err=$D/datagen.err 
  echo Logging to $log/$err
  time python -m CosmoSim --toml $D/dataset.toml --rnd \
         --csvfile $D/dataset.csv --outfile $D/roulette.csv  \
         --directory $D/images -vv  > $log 2> $err

fi

if test x$1 = xsetup
then
  time python -m droulette.split $D/problem.toml
fi


if test x$1 = xtrain
then
    shift
    for dir 
    do
       f=$dir/ml.toml
       echo $f
       log=$dir/modeltraining.log 
       err=$dir/modeltraining.err
       echo Logging to $log/$err
       time python -m droulette.model --config $f > $log 2> $err
    done
fi

if test x$1 = xeval
then

  log=$D/eval.log
  err=$D/eval.err 
  echo Logging to $log/$err

  time python -m droulette.eval -o $D/eval.csv $D/experiment??? > $log 2> $err
fi

echo Script completed
