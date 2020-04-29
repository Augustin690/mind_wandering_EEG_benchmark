featModelling <- function(subs, tasks, mtype, validType, f_parMat, gridSearchOn, searchBy, 
                          sessNormalize, get3bkData, 
                          contentsets = FALSE, taskMergeOn = FALSE,
                          parallelProcId = FALSE)
{
  
  # set pars based on inputs
  if (mtype == 'svm' && gridSearchOn){
    cPowerVec <- -5:15
    gPowerVec <- -15:3
  }
  
  ATPon <- length(tasks) == 2  && !taskMergeOn # do across-task pred only when two tasks are analyzed independently
  
  # intialize
  if (ATPon) {
    accMat <- matrix(0, length(subs), 4)
    colnames(accMat) <- c(tasks, paste0(tasks[1],'2',tasks[2]), paste0(tasks[2],'2',tasks[1]))
  } else if (!taskMergeOn) {
    accMat <- matrix(0, length(subs), length(tasks))
    colnames(accMat) <- c(tasks)
  } else {
    accMat <- matrix(0, length(subs), 1)
  }
  kapMat <- accMat; senMat <- accMat; speMat <- accMat
  
  if (mtype == 'svm' && gridSearchOn && !taskMergeOn) {
    parMat <- matrix(0, length(subs), length(tasks) * 2)
    colnames(parMat) <- paste0(rep(tasks, each = 2) , '_',  c('c', 'gamma'))
  } else if (mtype == 'svm' && gridSearchOn && taskMergeOn) {
    parMat <- matrix(0, length(subs), 2)
    colnames(parMat) <- c('c', 'gamma')
  } else if (mtype == 'svm' && !gridSearchOn && !is.nan(f_parMat)) {
    load(f_parMat)
    # check
    if (length(subs) != nrow(parMat)){
      stop('The specified parMat must match the participant number!')
    }
  } else {
    print('No parMat and grid search is off. Use default parameters to fit models!')
  }
  
  # set progress bar
  n <- length(subs) * length(tasks)
  i <- 0
  if (!is.numeric(parallelProcId)) {
    pb0 <- tkProgressBar(paste0(toupper(mtype),' learning progress'), min = 0, max = n, width = 300) 
  } else {
    pb0 <- tkProgressBar(paste0(toupper(mtype),' learning progress', parallelProcId), min = 0, max = n, width = 300) 
  }
  
  
  for (subi in 1:length(subs)){
    
    sub <- subs[subi]
    
    for (taski in 1:length(tasks)){
      
      if (taskMergeOn) {
        if (taski == 1) {tic(paste('model fitting sub', sub, 'task merged'))}
      } else {
        tic(paste('model fitting sub', sub, 'task', taski))
      }
      
      task       <- tasks[taski]
      if (ATPon) {testTask <- setdiff(tasks, task)}
      
      # load data
      if (get3bkData) {
        if (is.logical(contentsets) && !contentsets) {
          data <- get.data.3bk(sub, task,     measures, feats, folders)
          if (ATPon){test <- get.data.3bk(sub, testTask, measures, feats, folders)}
        } else {
          data <- get.data.content(sub, task, states, contentsets, measures, feats, folders, back3On = TRUE)
          if (ATPon){test <- get.data.content(sub, testTask, states, contentsets, measures, feats, folders, back3On = TRUE)}
        }
      } else if (!is.logical(contentsets)){
        data <- get.data.content(sub, task, states, contentsets, measures, feats, folders)
        if (ATPon){test <- get.data.content(sub, testTask, states, contentsets, measures, feats, folders)}
      } else {
        data <- get.data(sub, task,     measures, feats, folders, splitSessions = sessNormalize)
        if (ATPon){test <- get.data(sub, testTask, measures, feats, folders, splitSessions = sessNormalize)}
      } 
      
      # normalize
      if (sessNormalize) {
        
        for (sessi in 1:2){
          
          tempData <- normalize(data[[sessi]], 'z')$dataNorm
          if (ATPon){tempTest <- normalize(test[[sessi]], 'z')$dataNorm}
          
          if (sessi == 1){
            dataAll <- tempData
            if (ATPon) {testAll <- tempTest}
          } else {
            dataAll <- rbind(dataAll, tempData)
            if (ATPon) {testAll <- rbind(testAll, tempTest)}
          }
          
        }
        data <- dataAll  # the list turns to a df
        if (ATPon) {test <- testAll}
        
      } else {
        
        data <- normalize(data, 'z')$dataNorm
        if (ATPon) {test <- normalize(test, 'z')$dataNorm}
        
      }
      
      if (ATPon){
        # balance
        data.balan <- balance.class(data, 'copy') 
      }
      
      # taskMergeOn: catenate data
      if (taskMergeOn) {
        if (taski == 1) {
          data.merge <- data
        } else {
          data.merge <- rbind(data.merge, data)
        }
      }
      
      # do modelling: if taskMergeOn && taski < length(tasks), this part will be skipped
      if (!taskMergeOn | (taskMergeOn && taski == length(tasks))){
        
        # special care for task merged data
        if (taskMergeOn) {
          data <- data.merge
        }
        
        # grid search for the best pars
        if (mtype == 'svm' && gridSearchOn){
          mat <- grid.search.svm(data, 2^cPowerVec, 2^gPowerVec, validType = 'cv', searchBy = searchBy)
          
          # fail to do grid search?
          if (!is.matrix(mat) && is.nan(mat)){
            print('Fail to do grid search. Use default parameters to fit model!')
            if (taskMergeOn) {
              parMat[subi,] <- NaN
            } else {
              parMat[subi, c(1:2) + (taski - 1)*2] <- NaN
            }
            parList <- list()
          } else {
            pos <- which.localMax(mat, gPowerVec, cPowerVec)
            if (taskMergeOn) {
              parMat[subi,] <- 2^pos
              parList <- list(c = parMat[subi, 'c'], gamma = parMat[subi, 'gamma'])
            } else {
              parMat[subi, c(1:2) + (taski - 1)*2] <- 2^pos
              parList <- list(c = parMat[subi, paste0(task, '_c')], gamma = parMat[subi, paste0(task, '_gamma')])
            }
          }
          
        } else if (mtype == 'svm' && !is.nan(f_parMat)) { # use the specified pars
          parList <- list(c = parMat[subi, paste0(task, '_c')], gamma = parMat[subi, paste0(task, '_gamma')])
          if (is.nan(parList$c)){
            parList <- list()
          }
        } else { # use default pars
          parList <- list()
        }  # end of grid search and get the parList ready
        
        # train / test whinin tasks
        if (validType == 'loo'){
          perf <-    leave.one.out(data, mtype, parList = parList) # [default] balanced = 'copy' for the training split
        } else {
          perf <- cross.validation(data, mtype, parList = parList) # [default] balanced = 'copy' for the training split
        }
        
        # register performance
        if (!taskMergeOn) {
          colid <- taski
        } else {
          colid <- 1
        }
        accMat[subi, colid] <- perf$accuracy
        kapMat[subi, colid] <- perf$kappa
        senMat[subi, colid] <- perf$sensitivity
        speMat[subi, colid] <- perf$specificity
        
        # across-task prediction on: train / test between tasks
        if (ATPon){
          if (is.data.frame(data.balan)){
            m <- feat.modeling(data.balan, test, mtype, parList = parList)
            perf <- measure.performance(m$predictions, m$observations)
          } else if (is.nan(data.balan)) {
            perf <- list(accuracy = NaN, kappa = NaN, sensitivity = NaN, specificity = NaN)
          }
          accMat[subi, taski + 2] <- perf$accuracy
          kapMat[subi, taski + 2] <- perf$kappa
          senMat[subi, taski + 2] <- perf$sensitivity
          speMat[subi, taski + 2] <- perf$specificity
        }
        
      }  # end of grid search and modelling
      
      # save the temporary matrixes

      if (!is.numeric(parallelProcId)) {
        f_temp <- 'temp_featModelling.rdata'
      } else {
        f_temp <- paste0('temp_featModelling', parallelProcId, '.rdata')
      }

      
      print(paste0('Auto save as ', f_temp))
      if (mtype == 'svm' && gridSearchOn) {
        save(file = f_temp, accMat, kapMat, senMat, speMat, parMat)
      } else {
        save(file = f_temp, accMat, kapMat, senMat, speMat)
      }
      
      toc()
      
      i <- i+1
      setTkProgressBar(pb0, i, label = paste(round(i/n*100, 1), "% done")) 
      
    }  # loop over tasks
    
    #beep(8)
    
  }  # loop over subs
  close(pb0)
  
  return(f_temp)

}
