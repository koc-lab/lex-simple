output = "DiscourseSimplification/output_default.txt"
new_output = "uslt_final_output.txt"


f = open(output,"r")
fr = f.read()
outputs = fr.split("\n")
f.close()
refined_lines = []
"""
for line in outputs:
  flag_zero = 0
  for i in range(len(line)):
    #print(line[i-2:i]=="0\t")
    if line[i-3] == line[i-1] == "\t" and flag_zero == 0:
      print("yes")
      flag_zero = 1
      
      #new_line = line[i:]
            
      #for j in range(len(new_line)):
      
        #if new_line[j:j+6]=="L:LIST":
       
      refined_lines.append(line[i:])
      break


new_output = "newer_output.txt"
g = open(new_output,"w")
for line in refined_lines:
  g.write(line)
  if line != refined_lines[-1]:
    g.write("\n")
      
"""
counter = 0
flag = False
refined_lines = []
for line in outputs:
  if len(line) > 0:
    if line[0] == "#":
      if flag == True:
        refined_lines.append(sentence_lines)
      flag = True
      sentence_lines = []
      print("YES")
      counter += 1
      big_flag = True
      
  flag_zero = 0
  
  for i in range(len(line)):
    #print(line[i-2:i]=="0\t")
    
    #try:
    if line[i-3] == line[i-1] == "\t" and flag_zero == 0:
        print("yes")
        flag_zero = 1
        
        #new_line = line[i:]
              
        #for j in range(len(new_line)):
        
          #if new_line[j:j+6]=="L:LIST":
         
        sentence_lines.append(line[i:])
        break
    #except:
      #pass
refined_lines.append(sentence_lines)


g = open(new_output,"w")
for sent_lines in refined_lines:
  for line in sent_lines:
    g.write(line)
    if line[-1] != ".":
      g.write(".")
  if sent_lines != refined_lines[-1]:
    g.write("\n")

print(len(refined_lines))