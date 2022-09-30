 #block number >2
                    c=[]
                    for i in range( variable_number, variable_number + Aux +1):
                        if i != target:
                            c.append(i)
                    if ((depth - current_depth) % 2) == 1: 
                            #print(   "2n"   )   
                            #层数差 奇数 的存储位 为 variable_number +Aux- p+1+j  至 variable_number + Aux  要从差为偶数层 取数据
                            #层数差 偶数 的存储位 为 variable_number 至 variable_number + p -1      要从差为奇数层 取数据
                            #if block_number == 1:
                            #if p==2
                            #if block_number == 2:
                            
                            # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_number + Aux；另一个在 variable_number + Aux - block_number +2 上， variable_number + Aux - block_number +2 上放一个低一层的 clause。
                        if depth- current_depth == 1 : 
                        # UpPhase 1 升阶段 第一位要单独处理，其target 即最终target。控制位一个在variable_number + Aux；另一个在 variable_number + Aux - block_number +2 上， variable_number + Aux - block_number +2 上放一个低一层的 clause。
                        CCX | self._cgate([c[block_number-1 ] , c[2(block_number-1)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len -1 , c[block_number-1 ], current_depth-1, depth)
                        CCX | self._cgate([c[ block_number-1] , c[2(block_number-1)-1] , target])
                            
                            #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[0], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)
            
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        
                            #downPhase
                        for j in range(block_number-1 - 2 , 0, -1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])   

                        CCX | self._cgate([c[block_number-1 ] , c[2(block_number-1)-1] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID + block_len -1 , c[block_number-1 ], current_depth-1, depth)
                        CCX | self._cgate([c[ block_number-1] , c[2(block_number-1)-1] , target])

                            #repeat....

                            # 还原各个位置
                             #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[0], current_depth-1, depth)

                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[1], current_depth-1, depth)
            
                        CCX | self._cgate([c[1]  , c[0], c[ block_number ]])    
                        
                            #downPhase
                        for j in range(block_number-1 - 2 , 0, -1):
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 + (j+1)*block_len, c[(block_number-1)  -j], current_depth-1, depth)
                            CCX | self._cgate([c[(block_number-1)  -j] , c[2(block_number-1)-1  - j], c[2(block_number-1) - j]])   

                    else: #偶数层
                        CCX | self._cgate([c[Aux  -block_number ] , c[Aux - 2(block_number-1)] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID -1 + block_len , c[Aux  -block_number ], current_depth-1, depth)
                        CCX | self._cgate([c[Aux  -block_number] , c[Aux - 2(block_number-1)] , target])
                            
                            #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j*block_len , StartID -1 +(1+ j)*block_len, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[Aux-2], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])     
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1)*block_len, EndID, c[Aux-1], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])  
                        self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2)*block_len, StartID + (block_number-1)*block_len -1, c[Aux-2], current_depth-1, depth)
                        
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                            
                            #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range( EndID - StartID - 2, 0 , -1):
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])

                        CCX | self._cgate([c[Aux  -block_number ] , c[Aux - 2(block_number-1)] , target])
                        self.clause(CNF_data, variable_number, Aux, StartID, StartID , c[Aux  -block_number ], current_depth-1, depth)
                        CCX | self._cgate([c[Aux  -block_number] , c[Aux - 2(block_number-1)] , target])

                            # 还原各个位置

                            #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])
                                
                            # topPhase 
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[Aux-2], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])     
                        self.clause(CNF_data, variable_number, Aux, EndID, EndID, c[Aux-1], current_depth-1, depth)

                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])  
                        self.clause(CNF_data, variable_number, Aux, (EndID-1), EndID-1, c[Aux-2], current_depth-1, depth)
                        
                        CCX | self._cgate([c[Aux-2]  , c[Aux-1], c[ Aux-2 - EndID + StartID ]])    
                            
                            #控制位variable_number + Aux - (block_number-1) + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                        for j in range(1, block_number-1 - 1):
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])
                            self.clause(CNF_data, variable_number, Aux, StartID + j , StartID + j, c[Aux-(block_number-1)+j-1], current_depth-1, depth)
                            CCX | self._cgate([c[Aux-(block_number-1)+j-1] , c[Aux- 2(block_number-1) + j], c[Aux- 2(block_number-1) -1 + j]])





















































































































                            CCX | self._cgate([variable_number + Aux - block_number +2, variable_number + Aux, target])
                            self.clause(CNF_data, variable_number, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_number + Aux - block_number +2, current_depth-1, depth)
                            CCX | self._cgate([variable_number + Aux - block_number +2, variable_number + Aux , target])
                            
                            #控制位variable_number + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-2):
                                CCX | self._cgate([variable_number + Aux - block_number + 2 -j , variable_number + Aux -j, variable_number + Aux +1-j])
                                self.clause(CNF_data, variable_number, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_number + Aux - block_number + 2 -j, current_depth-1, depth)
                                CCX | self._cgate([variable_number + Aux - block_number + 2 -j , variable_number + Aux -j, variable_number + Aux +1-j])
                                    
                            # topPhase 

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + Aux - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_number + Aux - 2*block_number + 3, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + Aux - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])    
                        
                            #QuICT.qcda.synthesis.mct.

                            #downPhase
                            for j in range(1, block_number-2):
                                jdown = block_number-2 - j
                                CCX | self._cgate([variable_number + Aux - block_number + 2 - jdown, variable_number + Aux -jdown, variable_number + Aux +1-jdown] )
                                self.clause(CNF_data, variable_number, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_number + Aux - block_number + 2 -jdown, current_depth-1, depth)
                                CCX | self._cgate([variable_number + Aux - block_number + 2 -jdown , variable_number + Aux -jdown, variable_number + Aux +1-jdown] )
                            
                            CCX | self._cgate([variable_number + Aux - block_number +2, variable_number + Aux , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_number + Aux - block_number +2, current_depth-1, depth)
                            CCX | self._cgate([variable_number + Aux - block_number +2, variable_number + Aux , target])

                            #repeat....

                            # 还原各个位置

                            #控制位variable_number + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-2):
                                CCX | self._cgate([variable_number + Aux - block_number + 2 -j , variable_number + Aux -j, variable_number + Aux +1-j] )
                                self.clause(CNF_data, variable_number, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_number + Aux - block_number + 2 -j, current_depth-1, depth)
                                CCX | self._cgate([variable_number + Aux - block_number + 2 -j , variable_number + Aux -j, variable_number + Aux +1-j])
                                    
                            # topPhase 

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + Aux - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_number + Aux - 2*block_number + 3, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + Aux - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + Aux - 2*block_number + 4  , variable_number + Aux - 2*block_number + 3, variable_number + Aux - block_number + 3])    
                        
                            #downPhase
                            for j in range(1, block_number-2):
                                jdown = block_number-2 - j
                                CCX | self._cgate([variable_number + Aux - block_number + 2 - jdown, variable_number + Aux -jdown, variable_number + Aux +1-jdown] )
                                self.clause(CNF_data, variable_number, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_number + Aux - block_number + 2 -jdown, current_depth-1, depth)
                                CCX | self._cgate([variable_number + Aux - block_number + 2 -jdown , variable_number + Aux -jdown, variable_number + Aux +1-jdown] )
                            
                            # for j in range(block_number):
                            #    gate | Clause(CNF_data, variable_number, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_number + j, current_depth-1, depth)
                        else: #2n+1数层 处理
                            #print(   "2n+1"   )
                            CCX | self._cgate([variable_number + p-1 - block_number +2, variable_number + p-1 , target])
                            self.clause( CNF_data, variable_number, Aux, StartID, np.minimum(StartID + block_len - 1, EndID), variable_number + p-1 - block_number +2, current_depth-1, depth)
                            CCX | self._cgate([variable_number + p-1 - block_number +2, variable_number + p-1 , target])
                            
                            #控制位variable_number + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-2):
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 -j , variable_number + p-1 -j, variable_number + p -j] )
                                self.clause(CNF_data, variable_number, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_number + p-1 - block_number + 2 -j, current_depth-1, depth)
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 -j , variable_number + p-1 -j, variable_number + p -j] )
                                    
                            # topPhase 

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + p-1 - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_number + p-1 - 2*block_number + 3, current_depth-1, depth)

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + p-1 - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])    
                        
                            #QuICT.qcda.synthesis.mct.            

                            #downPhase
                            for j in range(1, block_number-2):
                                jdown = block_number-2 - j
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 - jdown, variable_number + p-1 -jdown, variable_number + p -jdown] )
                                self.clause(CNF_data, variable_number, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_number + p-1 - block_number + 2 -jdown, current_depth-1, depth)
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 -jdown , variable_number + p-1 -jdown, variable_number + p -jdown] )
                            
                            CCX | self._cgate([variable_number + p-1 - block_number +2, variable_number + p-1 , target])
                            self.clause(CNF_data, variable_number, Aux, StartID, np.minimum(StartID + block_len-1, EndID), variable_number + p-1 - block_number +2, current_depth-1, depth)
                            CCX | self._cgate([variable_number + p-1 - block_number +2, variable_number + p-1 , target])

                            #repeat....

                            # 还原各个位置

                            #控制位variable_number + Aux - block_number + 2 -j 放 clause ， 另一个 控制位(将与此for内的前一个target 相同)并和target 依次上升
                            for j in range(1, block_number-2):
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 -j , variable_number + p-1 -j, variable_number + p -j] )
                                self.clause(CNF_data, variable_number, Aux, StartID + j * block_len, np.minimum(StartID + j * block_len-1, EndID), variable_number + p-1 - block_number + 2 -j, current_depth-1, depth)
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 -j , variable_number + p-1 -j, variable_number + p -j] )
                                    
                            # topPhase 

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])    
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + p-1 - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-1) * block_len, np.minimum(StartID + (block_number) * block_len-1, EndID), variable_number + p-1 - 2*block_number + 3, current_depth-1, depth)

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])
                            self.clause(CNF_data, variable_number, Aux, StartID + (block_number-2) * block_len, np.minimum(StartID + (block_number-1) * block_len-1, EndID), variable_number + p-1 - 2*block_number + 4, current_depth-1, depth)

                            CCX | self._cgate([variable_number + p-1 - 2*block_number + 4  , variable_number + p-1 - 2*block_number + 3, variable_number + p-1 - block_number + 3])    
                    
                            #downPhase
                            for j in range(1, block_number-2):
                                jdown = block_number-2 - j
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 - jdown, variable_number + p-1 -jdown, variable_number + p -jdown] )
                                self.clause(CNF_data, variable_number, Aux, StartID + jdown * block_len, np.minimum(StartID + jdown * block_len-1, EndID), variable_number + p-1 - block_number + 2 -jdown, current_depth-1, depth)
                                CCX | self._cgate([variable_number + p-1 - block_number + 2 -jdown , variable_number + p-1 -jdown, variable_number + p -jdown] )
                