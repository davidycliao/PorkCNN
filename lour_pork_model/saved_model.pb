ä
ë¼
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68½

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
õÈ*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
õÈ*
dtype0
{
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*
shared_nameconv1d/kernel
t
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*#
_output_shapes
:Èd*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:d*
dtype0

conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd* 
shared_nameconv1d_1/kernel
x
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*#
_output_shapes
:Èd*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:d*
dtype0

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd* 
shared_nameconv1d_2/kernel
x
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*#
_output_shapes
:Èd*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:d*
dtype0

conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd* 
shared_nameconv1d_3/kernel
x
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*#
_output_shapes
:Èd*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:d*
dtype0

conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd* 
shared_nameconv1d_4/kernel
x
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*#
_output_shapes
:Èd*
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
:d*
dtype0

conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd* 
shared_nameconv1d_5/kernel
x
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*#
_output_shapes
:Èd*
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:d*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¼*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
¼*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
õÈ*,
shared_nameAdam/embedding/embeddings/m

/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m* 
_output_shapes
:
õÈ*
dtype0

Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*#
_output_shapes
:Èd*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:d*
dtype0

Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_1/kernel/m

*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:d*
dtype0

Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_2/kernel/m

*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
:d*
dtype0

Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_3/kernel/m

*Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/m*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_3/bias/m
y
(Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/m*
_output_shapes
:d*
dtype0

Adam/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_4/kernel/m

*Adam/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/kernel/m*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_4/bias/m
y
(Adam/conv1d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/bias/m*
_output_shapes
:d*
dtype0

Adam/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_5/kernel/m

*Adam/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/kernel/m*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_5/bias/m
y
(Adam/conv1d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/bias/m*
_output_shapes
:d*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¼*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
¼*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
õÈ*,
shared_nameAdam/embedding/embeddings/v

/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v* 
_output_shapes
:
õÈ*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*#
_output_shapes
:Èd*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:d*
dtype0

Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_1/kernel/v

*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:d*
dtype0

Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_2/kernel/v

*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
:d*
dtype0

Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_3/kernel/v

*Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/v*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_3/bias/v
y
(Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/v*
_output_shapes
:d*
dtype0

Adam/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_4/kernel/v

*Adam/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/kernel/v*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_4/bias/v
y
(Adam/conv1d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/bias/v*
_output_shapes
:d*
dtype0

Adam/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Èd*'
shared_nameAdam/conv1d_5/kernel/v

*Adam/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/kernel/v*#
_output_shapes
:Èd*
dtype0

Adam/conv1d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv1d_5/bias/v
y
(Adam/conv1d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/bias/v*
_output_shapes
:d*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¼*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
¼*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ãj
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*þi
valueôiBñi Bêi
ß
	embedding

bigram
bigram2
trigram
trigram2
fourgram
fivegram
pool
	dense_1

dropout

last_dense
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
¦

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
¦

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
¦

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
¦

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
¦

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*

L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
¦

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
¥
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses* 
¦

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*

iiter

jbeta_1

kbeta_2
	ldecay
mlearning_ratem¶m·m¸$m¹%mº,m»-m¼4m½5m¾<m¿=mÀDmÁEmÂRmÃSmÄamÅbmÆvÇvÈvÉ$vÊ%vË,vÌ-vÍ4vÎ5vÏ<vÐ=vÑDvÒEvÓRvÔSvÕavÖbv×*

0
1
2
$3
%4
,5
-6
47
58
<9
=10
D11
E12
R13
S14
a15
b16*

0
1
2
$3
%4
,5
-6
47
58
<9
=10
D11
E12
R13
S14
a15
b16*
* 
°
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

sserving_default* 
]W
VARIABLE_VALUEembedding/embeddings/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUEconv1d/kernel(bigram/kernel/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv1d/bias&bigram/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
RL
VARIABLE_VALUEconv1d_1/kernel)bigram2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv1d_1/bias'bigram2/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
RL
VARIABLE_VALUEconv1d_2/kernel)trigram/kernel/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv1d_2/bias'trigram/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
SM
VARIABLE_VALUEconv1d_3/kernel*trigram2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv1d_3/bias(trigram2/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
SM
VARIABLE_VALUEconv1d_4/kernel*fourgram/kernel/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv1d_4/bias(fourgram/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
SM
VARIABLE_VALUEconv1d_5/kernel*fivegram/kernel/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv1d_5/bias(fivegram/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 
* 
* 
OI
VARIABLE_VALUEdense/kernel)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUE*

R0
S1*

R0
S1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 
* 
* 
* 
TN
VARIABLE_VALUEdense_1/kernel,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_1/bias*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

a0
b1*
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

«0
¬1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

­total

®count
¯	variables
°	keras_api*
M

±total

²count
³
_fn_kwargs
´	variables
µ	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

­0
®1*

¯	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

±0
²1*

´	variables*
z
VARIABLE_VALUEAdam/embedding/embeddings/mKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d/kernel/mDbigram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv1d/bias/mBbigram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv1d_1/kernel/mEbigram2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_1/bias/mCbigram2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv1d_2/kernel/mEtrigram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_2/bias/mCtrigram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv1d_3/kernel/mFtrigram2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d_3/bias/mDtrigram2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv1d_4/kernel/mFfourgram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d_4/bias/mDfourgram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv1d_5/kernel/mFfivegram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d_5/bias/mDfivegram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense/kernel/mEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/mCdense_1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/dense_1/kernel/mHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_1/bias/mFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/embedding/embeddings/vKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d/kernel/vDbigram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv1d/bias/vBbigram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv1d_1/kernel/vEbigram2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_1/bias/vCbigram2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv1d_2/kernel/vEtrigram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_2/bias/vCtrigram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv1d_3/kernel/vFtrigram2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d_3/bias/vDtrigram2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv1d_4/kernel/vFfourgram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d_4/bias/vDfourgram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv1d_5/kernel/vFfivegram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv1d_5/bias/vDfivegram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense/kernel/vEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/vCdense_1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/dense_1/kernel/vHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_1/bias/vFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1embedding/embeddingsconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_221372
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ø
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp*Adam/conv1d_3/kernel/m/Read/ReadVariableOp(Adam/conv1d_3/bias/m/Read/ReadVariableOp*Adam/conv1d_4/kernel/m/Read/ReadVariableOp(Adam/conv1d_4/bias/m/Read/ReadVariableOp*Adam/conv1d_5/kernel/m/Read/ReadVariableOp(Adam/conv1d_5/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp*Adam/conv1d_3/kernel/v/Read/ReadVariableOp(Adam/conv1d_3/bias/v/Read/ReadVariableOp*Adam/conv1d_4/kernel/v/Read/ReadVariableOp(Adam/conv1d_4/bias/v/Read/ReadVariableOp*Adam/conv1d_5/kernel/v/Read/ReadVariableOp(Adam/conv1d_5/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*I
TinB
@2>	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_221819
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/conv1d_3/kernel/mAdam/conv1d_3/bias/mAdam/conv1d_4/kernel/mAdam/conv1d_4/bias/mAdam/conv1d_5/kernel/mAdam/conv1d_5/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/embedding/embeddings/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/conv1d_3/kernel/vAdam/conv1d_3/bias/vAdam/conv1d_4/kernel/vAdam/conv1d_4/bias/vAdam/conv1d_5/kernel/vAdam/conv1d_5/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_222009½·
ñ
à
C__inference_PorkCNN_layer_call_and_return_conditional_losses_221331

inputs5
!embedding_embedding_lookup_221205:
õÈI
2conv1d_conv1d_expanddims_1_readvariableop_resource:Èd4
&conv1d_biasadd_readvariableop_resource:dK
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_1_biasadd_readvariableop_resource:dK
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_2_biasadd_readvariableop_resource:dK
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_3_biasadd_readvariableop_resource:dK
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_4_biasadd_readvariableop_resource:dK
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_5_biasadd_readvariableop_resource:d8
$dense_matmul_readvariableop_resource:
¼4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_3/BiasAdd/ReadVariableOp¢+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_4/BiasAdd/ReadVariableOp¢!conv1d_4/BiasAdd_1/ReadVariableOp¢+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp¢conv1d_5/BiasAdd/ReadVariableOp¢+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢embedding/embedding_lookupÙ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_221205inputs*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/221205*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0Â
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/221205*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¹
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¡
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¶
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÃ
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¡
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_1/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_1Maxconv1d_1/Relu:activations:05global_max_pooling1d/Max_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_2/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_2Maxconv1d_2/Relu:activations:05global_max_pooling1d/Max_2/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_3/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_3Maxconv1d_3/Relu:activations:05global_max_pooling1d/Max_3/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_4/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_4Maxconv1d_4/Relu:activations:05global_max_pooling1d/Max_4/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
 conv1d_4/Conv1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÁ
conv1d_4/Conv1D_1/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0)conv1d_4/Conv1D_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ§
-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0d
"conv1d_4/Conv1D_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Â
conv1d_4/Conv1D_1/ExpandDims_1
ExpandDims5conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp:value:0+conv1d_4/Conv1D_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÏ
conv1d_4/Conv1D_1Conv2D%conv1d_4/Conv1D_1/ExpandDims:output:0'conv1d_4/Conv1D_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_4/Conv1D_1/SqueezeSqueezeconv1d_4/Conv1D_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
!conv1d_4/BiasAdd_1/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0£
conv1d_4/BiasAdd_1BiasAdd"conv1d_4/Conv1D_1/Squeeze:output:0)conv1d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
conv1d_4/Relu_1Reluconv1d_4/BiasAdd_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :©
global_max_pooling1d/Max_5Maxconv1d_4/Relu_1:activations:05global_max_pooling1d/Max_5/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_5/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_6Maxconv1d_5/Relu:activations:05global_max_pooling1d/Max_6/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿå
concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d/Max_1:output:0#global_max_pooling1d/Max_2:output:0#global_max_pooling1d/Max_3:output:0#global_max_pooling1d/Max_4:output:0#global_max_pooling1d/Max_5:output:0#global_max_pooling1d/Max_6:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
¼*
dtype0
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >¿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp"^conv1d_4/BiasAdd_1/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp.^conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2F
!conv1d_4/BiasAdd_1/ReadVariableOp!conv1d_4/BiasAdd_1/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2^
-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_1_layer_call_and_return_conditional_losses_221438

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ñ
a
(__inference_dropout_layer_call_fn_221579

inputs
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_220618p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_4_layer_call_and_return_conditional_losses_221513

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ô

B__inference_conv1d_layer_call_and_return_conditional_losses_221413

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ú
a
C__inference_dropout_layer_call_and_return_conditional_losses_221584

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ö
(__inference_PorkCNN_layer_call_fn_221080

inputs
unknown:
õÈ 
	unknown_0:Èd
	unknown_1:d 
	unknown_2:Èd
	unknown_3:d 
	unknown_4:Èd
	unknown_5:d 
	unknown_6:Èd
	unknown_7:d 
	unknown_8:Èd
	unknown_9:d!

unknown_10:Èd

unknown_11:d

unknown_12:
¼

unknown_13:	

unknown_14:	

unknown_15:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220800o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
N

C__inference_PorkCNN_layer_call_and_return_conditional_losses_220551

inputs$
embedding_220362:
õÈ$
conv1d_220382:Èd
conv1d_220384:d&
conv1d_1_220405:Èd
conv1d_1_220407:d&
conv1d_2_220428:Èd
conv1d_2_220430:d&
conv1d_3_220451:Èd
conv1d_3_220453:d&
conv1d_4_220474:Èd
conv1d_4_220476:d&
conv1d_5_220501:Èd
conv1d_5_220503:d 
dense_220521:
¼
dense_220523:	!
dense_1_220545:	
dense_1_220547:
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢"conv1d_4/StatefulPartitionedCall_1¢ conv1d_5/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!embedding/StatefulPartitionedCallæ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_220362*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_220361
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_220382conv1d_220384*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_220381ñ
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_220405conv1d_1_220407*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_220404õ
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_220428conv1d_2_220430*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_220427õ
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3_220451conv1d_3_220453*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_220450õ
&global_max_pooling1d/PartitionedCall_3PartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220474conv1d_4_220476*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473õ
&global_max_pooling1d/PartitionedCall_4PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
"conv1d_4/StatefulPartitionedCall_1StatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220474conv1d_4_220476*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473÷
&global_max_pooling1d/PartitionedCall_5PartitionedCall+conv1d_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_5_220501conv1d_5_220503*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_220500õ
&global_max_pooling1d/PartitionedCall_6PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¹
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0/global_max_pooling1d/PartitionedCall_3:output:0/global_max_pooling1d/PartitionedCall_4:output:0/global_max_pooling1d/PartitionedCall_5:output:0/global_max_pooling1d/PartitionedCall_6:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼î
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_220521dense_220523*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220520×
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_220531
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_220545dense_1_220547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220544w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall#^conv1d_4/StatefulPartitionedCall_1!^conv1d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2H
"conv1d_4/StatefulPartitionedCall_1"conv1d_4/StatefulPartitionedCall_12D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½O
¬
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220996
input_1$
embedding_220939:
õÈ$
conv1d_220942:Èd
conv1d_220944:d&
conv1d_1_220948:Èd
conv1d_1_220950:d&
conv1d_2_220954:Èd
conv1d_2_220956:d&
conv1d_3_220960:Èd
conv1d_3_220962:d&
conv1d_4_220966:Èd
conv1d_4_220968:d&
conv1d_5_220976:Èd
conv1d_5_220978:d 
dense_220984:
¼
dense_220986:	!
dense_1_220990:	
dense_1_220992:
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢"conv1d_4/StatefulPartitionedCall_1¢ conv1d_5/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!embedding/StatefulPartitionedCallç
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_220939*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_220361
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_220942conv1d_220944*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_220381ñ
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_220948conv1d_1_220950*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_220404õ
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_220954conv1d_2_220956*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_220427õ
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3_220960conv1d_3_220962*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_220450õ
&global_max_pooling1d/PartitionedCall_3PartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220966conv1d_4_220968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473õ
&global_max_pooling1d/PartitionedCall_4PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
"conv1d_4/StatefulPartitionedCall_1StatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220966conv1d_4_220968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473÷
&global_max_pooling1d/PartitionedCall_5PartitionedCall+conv1d_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_5_220976conv1d_5_220978*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_220500õ
&global_max_pooling1d/PartitionedCall_6PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¹
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0/global_max_pooling1d/PartitionedCall_3:output:0/global_max_pooling1d/PartitionedCall_4:output:0/global_max_pooling1d/PartitionedCall_5:output:0/global_max_pooling1d/PartitionedCall_6:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼î
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_220984dense_220986*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220520ç
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_220618
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_220990dense_1_220992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220544w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall#^conv1d_4/StatefulPartitionedCall_1!^conv1d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2H
"conv1d_4/StatefulPartitionedCall_1"conv1d_4/StatefulPartitionedCall_12D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
í
×
(__inference_PorkCNN_layer_call_fn_220876
input_1
unknown:
õÈ 
	unknown_0:Èd
	unknown_1:d 
	unknown_2:Èd
	unknown_3:d 
	unknown_4:Èd
	unknown_5:d 
	unknown_6:Èd
	unknown_7:d 
	unknown_8:Èd
	unknown_9:d!

unknown_10:Èd

unknown_11:d

unknown_12:
¼

unknown_13:	

unknown_14:	

unknown_15:
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220800o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Îé
ö$
"__inference__traced_restore_222009
file_prefix9
%assignvariableop_embedding_embeddings:
õÈ7
 assignvariableop_1_conv1d_kernel:Èd,
assignvariableop_2_conv1d_bias:d9
"assignvariableop_3_conv1d_1_kernel:Èd.
 assignvariableop_4_conv1d_1_bias:d9
"assignvariableop_5_conv1d_2_kernel:Èd.
 assignvariableop_6_conv1d_2_bias:d9
"assignvariableop_7_conv1d_3_kernel:Èd.
 assignvariableop_8_conv1d_3_bias:d9
"assignvariableop_9_conv1d_4_kernel:Èd/
!assignvariableop_10_conv1d_4_bias:d:
#assignvariableop_11_conv1d_5_kernel:Èd/
!assignvariableop_12_conv1d_5_bias:d4
 assignvariableop_13_dense_kernel:
¼-
assignvariableop_14_dense_bias:	5
"assignvariableop_15_dense_1_kernel:	.
 assignvariableop_16_dense_1_bias:'
assignvariableop_17_adam_iter:	 )
assignvariableop_18_adam_beta_1: )
assignvariableop_19_adam_beta_2: (
assignvariableop_20_adam_decay: 0
&assignvariableop_21_adam_learning_rate: #
assignvariableop_22_total: #
assignvariableop_23_count: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: C
/assignvariableop_26_adam_embedding_embeddings_m:
õÈ?
(assignvariableop_27_adam_conv1d_kernel_m:Èd4
&assignvariableop_28_adam_conv1d_bias_m:dA
*assignvariableop_29_adam_conv1d_1_kernel_m:Èd6
(assignvariableop_30_adam_conv1d_1_bias_m:dA
*assignvariableop_31_adam_conv1d_2_kernel_m:Èd6
(assignvariableop_32_adam_conv1d_2_bias_m:dA
*assignvariableop_33_adam_conv1d_3_kernel_m:Èd6
(assignvariableop_34_adam_conv1d_3_bias_m:dA
*assignvariableop_35_adam_conv1d_4_kernel_m:Èd6
(assignvariableop_36_adam_conv1d_4_bias_m:dA
*assignvariableop_37_adam_conv1d_5_kernel_m:Èd6
(assignvariableop_38_adam_conv1d_5_bias_m:d;
'assignvariableop_39_adam_dense_kernel_m:
¼4
%assignvariableop_40_adam_dense_bias_m:	<
)assignvariableop_41_adam_dense_1_kernel_m:	5
'assignvariableop_42_adam_dense_1_bias_m:C
/assignvariableop_43_adam_embedding_embeddings_v:
õÈ?
(assignvariableop_44_adam_conv1d_kernel_v:Èd4
&assignvariableop_45_adam_conv1d_bias_v:dA
*assignvariableop_46_adam_conv1d_1_kernel_v:Èd6
(assignvariableop_47_adam_conv1d_1_bias_v:dA
*assignvariableop_48_adam_conv1d_2_kernel_v:Èd6
(assignvariableop_49_adam_conv1d_2_bias_v:dA
*assignvariableop_50_adam_conv1d_3_kernel_v:Èd6
(assignvariableop_51_adam_conv1d_3_bias_v:dA
*assignvariableop_52_adam_conv1d_4_kernel_v:Èd6
(assignvariableop_53_adam_conv1d_4_bias_v:dA
*assignvariableop_54_adam_conv1d_5_kernel_v:Èd6
(assignvariableop_55_adam_conv1d_5_bias_v:d;
'assignvariableop_56_adam_dense_kernel_v:
¼4
%assignvariableop_57_adam_dense_bias_v:	<
)assignvariableop_58_adam_dense_1_kernel_v:	5
'assignvariableop_59_adam_dense_1_bias_v:
identity_61¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*¹
value¯B¬=B/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB(bigram/kernel/.ATTRIBUTES/VARIABLE_VALUEB&bigram/bias/.ATTRIBUTES/VARIABLE_VALUEB)bigram2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'bigram2/bias/.ATTRIBUTES/VARIABLE_VALUEB)trigram/kernel/.ATTRIBUTES/VARIABLE_VALUEB'trigram/bias/.ATTRIBUTES/VARIABLE_VALUEB*trigram2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(trigram2/bias/.ATTRIBUTES/VARIABLE_VALUEB*fourgram/kernel/.ATTRIBUTES/VARIABLE_VALUEB(fourgram/bias/.ATTRIBUTES/VARIABLE_VALUEB*fivegram/kernel/.ATTRIBUTES/VARIABLE_VALUEB(fivegram/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDbigram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBbigram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEbigram2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCbigram2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEtrigram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCtrigram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFtrigram2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDtrigram2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFfourgram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDfourgram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFfivegram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDfivegram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDbigram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBbigram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEbigram2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCbigram2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEtrigram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCtrigram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFtrigram2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDtrigram2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFfourgram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDfourgram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFfivegram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDfivegram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHí
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ò
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes÷
ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv1d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv1d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv1d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv1d_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv1d_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv1d_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv1d_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv1d_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv1d_5_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv1d_5_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_dense_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_iterIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_embedding_embeddings_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv1d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv1d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv1d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv1d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv1d_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv1d_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_5_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_5_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_43AssignVariableOp/assignvariableop_43_adam_embedding_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv1d_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_conv1d_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_conv1d_1_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv1d_1_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_conv1d_2_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv1d_2_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_conv1d_3_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_conv1d_3_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_conv1d_4_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv1d_4_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv1d_5_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv1d_5_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp%assignvariableop_57_adam_dense_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_dense_1_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_dense_1_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ÷

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: ä

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ö

D__inference_conv1d_5_layer_call_and_return_conditional_losses_220500

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ºO
«
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220800

inputs$
embedding_220743:
õÈ$
conv1d_220746:Èd
conv1d_220748:d&
conv1d_1_220752:Èd
conv1d_1_220754:d&
conv1d_2_220758:Èd
conv1d_2_220760:d&
conv1d_3_220764:Èd
conv1d_3_220766:d&
conv1d_4_220770:Èd
conv1d_4_220772:d&
conv1d_5_220780:Èd
conv1d_5_220782:d 
dense_220788:
¼
dense_220790:	!
dense_1_220794:	
dense_1_220796:
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢"conv1d_4/StatefulPartitionedCall_1¢ conv1d_5/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!embedding/StatefulPartitionedCallæ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_220743*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_220361
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_220746conv1d_220748*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_220381ñ
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_220752conv1d_1_220754*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_220404õ
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_220758conv1d_2_220760*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_220427õ
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3_220764conv1d_3_220766*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_220450õ
&global_max_pooling1d/PartitionedCall_3PartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220770conv1d_4_220772*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473õ
&global_max_pooling1d/PartitionedCall_4PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
"conv1d_4/StatefulPartitionedCall_1StatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220770conv1d_4_220772*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473÷
&global_max_pooling1d/PartitionedCall_5PartitionedCall+conv1d_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_5_220780conv1d_5_220782*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_220500õ
&global_max_pooling1d/PartitionedCall_6PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¹
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0/global_max_pooling1d/PartitionedCall_3:output:0/global_max_pooling1d/PartitionedCall_4:output:0/global_max_pooling1d/PartitionedCall_5:output:0/global_max_pooling1d/PartitionedCall_6:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼î
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_220788dense_220790*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220520ç
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_220618
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_220794dense_1_220796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220544w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall#^conv1d_4/StatefulPartitionedCall_1!^conv1d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2H
"conv1d_4/StatefulPartitionedCall_1"conv1d_4/StatefulPartitionedCall_12D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_221549

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
Q
5__inference_global_max_pooling1d_layer_call_fn_221543

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

(__inference_dense_1_layer_call_fn_221605

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
¤
E__inference_embedding_layer_call_and_return_conditional_losses_221388

inputs+
embedding_lookup_221382:
õÈ
identity¢embedding_lookup»
embedding_lookupResourceGatherembedding_lookup_221382inputs*
Tindices0**
_class 
loc:@embedding_lookup/221382*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/221382*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

)__inference_conv1d_3_layer_call_fn_221472

inputs
unknown:Èd
	unknown_0:d
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_220450t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs


õ
C__inference_dense_1_layer_call_and_return_conditional_losses_221616

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
¤
E__inference_embedding_layer_call_and_return_conditional_losses_220361

inputs+
embedding_lookup_220355:
õÈ
identity¢embedding_lookup»
embedding_lookupResourceGatherembedding_lookup_220355inputs*
Tindices0**
_class 
loc:@embedding_lookup/220355*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/220355*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
à
C__inference_PorkCNN_layer_call_and_return_conditional_losses_221202

inputs5
!embedding_embedding_lookup_221083:
õÈI
2conv1d_conv1d_expanddims_1_readvariableop_resource:Èd4
&conv1d_biasadd_readvariableop_resource:dK
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_1_biasadd_readvariableop_resource:dK
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_2_biasadd_readvariableop_resource:dK
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_3_biasadd_readvariableop_resource:dK
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_4_biasadd_readvariableop_resource:dK
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:Èd6
(conv1d_5_biasadd_readvariableop_resource:d8
$dense_matmul_readvariableop_resource:
¼4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_3/BiasAdd/ReadVariableOp¢+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_4/BiasAdd/ReadVariableOp¢!conv1d_4/BiasAdd_1/ReadVariableOp¢+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp¢conv1d_5/BiasAdd/ReadVariableOp¢+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢embedding/embedding_lookupÙ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_221083inputs*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/221083*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0Â
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/221083*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¹
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¡
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¶
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÃ
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¡
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_1/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_1Maxconv1d_1/Relu:activations:05global_max_pooling1d/Max_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_2/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_2Maxconv1d_2/Relu:activations:05global_max_pooling1d/Max_2/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_3/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_3Maxconv1d_3/Relu:activations:05global_max_pooling1d/Max_3/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_4/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_4Maxconv1d_4/Relu:activations:05global_max_pooling1d/Max_4/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
 conv1d_4/Conv1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÁ
conv1d_4/Conv1D_1/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0)conv1d_4/Conv1D_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ§
-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0d
"conv1d_4/Conv1D_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Â
conv1d_4/Conv1D_1/ExpandDims_1
ExpandDims5conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp:value:0+conv1d_4/Conv1D_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÏ
conv1d_4/Conv1D_1Conv2D%conv1d_4/Conv1D_1/ExpandDims:output:0'conv1d_4/Conv1D_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_4/Conv1D_1/SqueezeSqueezeconv1d_4/Conv1D_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
!conv1d_4/BiasAdd_1/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0£
conv1d_4/BiasAdd_1BiasAdd"conv1d_4/Conv1D_1/Squeeze:output:0)conv1d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
conv1d_4/Relu_1Reluconv1d_4/BiasAdd_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :©
global_max_pooling1d/Max_5Maxconv1d_4/Relu_1:activations:05global_max_pooling1d/Max_5/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ½
conv1d_5/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¥
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¼
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÉ
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
,global_max_pooling1d/Max_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :§
global_max_pooling1d/Max_6Maxconv1d_5/Relu:activations:05global_max_pooling1d/Max_6/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿå
concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d/Max_1:output:0#global_max_pooling1d/Max_2:output:0#global_max_pooling1d/Max_3:output:0#global_max_pooling1d/Max_4:output:0#global_max_pooling1d/Max_5:output:0#global_max_pooling1d/Max_6:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
¼*
dtype0
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp"^conv1d_4/BiasAdd_1/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp.^conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2F
!conv1d_4/BiasAdd_1/ReadVariableOp!conv1d_4/BiasAdd_1/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2^
-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp-conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
N

C__inference_PorkCNN_layer_call_and_return_conditional_losses_220936
input_1$
embedding_220879:
õÈ$
conv1d_220882:Èd
conv1d_220884:d&
conv1d_1_220888:Èd
conv1d_1_220890:d&
conv1d_2_220894:Èd
conv1d_2_220896:d&
conv1d_3_220900:Èd
conv1d_3_220902:d&
conv1d_4_220906:Èd
conv1d_4_220908:d&
conv1d_5_220916:Èd
conv1d_5_220918:d 
dense_220924:
¼
dense_220926:	!
dense_1_220930:	
dense_1_220932:
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢"conv1d_4/StatefulPartitionedCall_1¢ conv1d_5/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!embedding/StatefulPartitionedCallç
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_220879*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_220361
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_220882conv1d_220884*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_220381ñ
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_220888conv1d_1_220890*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_220404õ
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_220894conv1d_2_220896*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_220427õ
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3_220900conv1d_3_220902*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_220450õ
&global_max_pooling1d/PartitionedCall_3PartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220906conv1d_4_220908*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473õ
&global_max_pooling1d/PartitionedCall_4PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
"conv1d_4/StatefulPartitionedCall_1StatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_4_220906conv1d_4_220908*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473÷
&global_max_pooling1d/PartitionedCall_5PartitionedCall+conv1d_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_5_220916conv1d_5_220918*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_220500õ
&global_max_pooling1d/PartitionedCall_6PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¹
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0/global_max_pooling1d/PartitionedCall_3:output:0/global_max_pooling1d/PartitionedCall_4:output:0/global_max_pooling1d/PartitionedCall_5:output:0/global_max_pooling1d/PartitionedCall_6:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼î
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_220924dense_220926*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220520×
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_220531
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_220930dense_1_220932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220544w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall#^conv1d_4/StatefulPartitionedCall_1!^conv1d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2H
"conv1d_4/StatefulPartitionedCall_1"conv1d_4/StatefulPartitionedCall_12D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_220342

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_3_layer_call_and_return_conditional_losses_221488

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_2_layer_call_and_return_conditional_losses_220427

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ý

)__inference_conv1d_2_layer_call_fn_221447

inputs
unknown:Èd
	unknown_0:d
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_220427t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_3_layer_call_and_return_conditional_losses_220450

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¤

õ
A__inference_dense_layer_call_and_return_conditional_losses_220520

inputs2
matmul_readvariableop_resource:
¼.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¼*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¼: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 
_user_specified_nameinputs
Ç
Ó
$__inference_signature_wrapper_221372
input_1
unknown:
õÈ 
	unknown_0:Èd
	unknown_1:d 
	unknown_2:Èd
	unknown_3:d 
	unknown_4:Èd
	unknown_5:d 
	unknown_6:Èd
	unknown_7:d 
	unknown_8:Èd
	unknown_9:d!

unknown_10:Èd

unknown_11:d

unknown_12:
¼

unknown_13:	

unknown_14:	

unknown_15:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_220332o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ý

)__inference_conv1d_1_layer_call_fn_221422

inputs
unknown:Èd
	unknown_0:d
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_220404t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_2_layer_call_and_return_conditional_losses_221463

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ý

)__inference_conv1d_4_layer_call_fn_221497

inputs
unknown:Èd
	unknown_0:d
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¤

õ
A__inference_dense_layer_call_and_return_conditional_losses_221569

inputs2
matmul_readvariableop_resource:
¼.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¼*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¼: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 
_user_specified_nameinputs
Ö

D__inference_conv1d_5_layer_call_and_return_conditional_losses_221538

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ý

)__inference_conv1d_5_layer_call_fn_221522

inputs
unknown:Èd
	unknown_0:d
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_220500t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_4_layer_call_and_return_conditional_losses_220473

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

D
(__inference_dropout_layer_call_fn_221574

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_220531a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

B__inference_conv1d_layer_call_and_return_conditional_losses_220381

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ã

&__inference_dense_layer_call_fn_221558

inputs
unknown:
¼
	unknown_0:	
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220520p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¼: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 
_user_specified_nameinputs
q
¡
__inference__traced_save_221819
file_prefix3
/savev2_embedding_embeddings_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_4_kernel_m_read_readvariableop3
/savev2_adam_conv1d_4_bias_m_read_readvariableop5
1savev2_adam_conv1d_5_kernel_m_read_readvariableop3
/savev2_adam_conv1d_5_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop5
1savev2_adam_conv1d_4_kernel_v_read_readvariableop3
/savev2_adam_conv1d_4_bias_v_read_readvariableop5
1savev2_adam_conv1d_5_kernel_v_read_readvariableop3
/savev2_adam_conv1d_5_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*¹
value¯B¬=B/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB(bigram/kernel/.ATTRIBUTES/VARIABLE_VALUEB&bigram/bias/.ATTRIBUTES/VARIABLE_VALUEB)bigram2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'bigram2/bias/.ATTRIBUTES/VARIABLE_VALUEB)trigram/kernel/.ATTRIBUTES/VARIABLE_VALUEB'trigram/bias/.ATTRIBUTES/VARIABLE_VALUEB*trigram2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(trigram2/bias/.ATTRIBUTES/VARIABLE_VALUEB*fourgram/kernel/.ATTRIBUTES/VARIABLE_VALUEB(fourgram/bias/.ATTRIBUTES/VARIABLE_VALUEB*fivegram/kernel/.ATTRIBUTES/VARIABLE_VALUEB(fivegram/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDbigram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBbigram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEbigram2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCbigram2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEtrigram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCtrigram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFtrigram2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDtrigram2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFfourgram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDfourgram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFfivegram/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDfivegram/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDbigram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBbigram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEbigram2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCbigram2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEtrigram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCtrigram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFtrigram2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDtrigram2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFfourgram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDfourgram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFfivegram/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDfivegram/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHê
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ±
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop1savev2_adam_conv1d_5_kernel_m_read_readvariableop/savev2_adam_conv1d_5_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableop1savev2_adam_conv1d_5_kernel_v_read_readvariableop/savev2_adam_conv1d_5_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*µ
_input_shapes£
 : :
õÈ:Èd:d:Èd:d:Èd:d:Èd:d:Èd:d:Èd:d:
¼::	:: : : : : : : : : :
õÈ:Èd:d:Èd:d:Èd:d:Èd:d:Èd:d:Èd:d:
¼::	::
õÈ:Èd:d:Èd:d:Èd:d:Èd:d:Èd:d:Èd:d:
¼::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
õÈ:)%
#
_output_shapes
:Èd: 

_output_shapes
:d:)%
#
_output_shapes
:Èd: 

_output_shapes
:d:)%
#
_output_shapes
:Èd: 

_output_shapes
:d:)%
#
_output_shapes
:Èd: 	

_output_shapes
:d:)
%
#
_output_shapes
:Èd: 

_output_shapes
:d:)%
#
_output_shapes
:Èd: 

_output_shapes
:d:&"
 
_output_shapes
:
¼:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
õÈ:)%
#
_output_shapes
:Èd: 

_output_shapes
:d:)%
#
_output_shapes
:Èd: 

_output_shapes
:d:) %
#
_output_shapes
:Èd: !

_output_shapes
:d:)"%
#
_output_shapes
:Èd: #

_output_shapes
:d:)$%
#
_output_shapes
:Èd: %

_output_shapes
:d:)&%
#
_output_shapes
:Èd: '

_output_shapes
:d:&("
 
_output_shapes
:
¼:!)

_output_shapes	
::%*!

_output_shapes
:	: +

_output_shapes
::&,"
 
_output_shapes
:
õÈ:)-%
#
_output_shapes
:Èd: .

_output_shapes
:d:)/%
#
_output_shapes
:Èd: 0

_output_shapes
:d:)1%
#
_output_shapes
:Èd: 2

_output_shapes
:d:)3%
#
_output_shapes
:Èd: 4

_output_shapes
:d:)5%
#
_output_shapes
:Èd: 6

_output_shapes
:d:)7%
#
_output_shapes
:Èd: 8

_output_shapes
:d:&9"
 
_output_shapes
:
¼:!:

_output_shapes	
::%;!

_output_shapes
:	: <

_output_shapes
::=

_output_shapes
: 
ù	
b
C__inference_dropout_layer_call_and_return_conditional_losses_221596

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬

*__inference_embedding_layer_call_fn_221379

inputs
unknown:
õÈ
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_220361u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
×
(__inference_PorkCNN_layer_call_fn_220588
input_1
unknown:
õÈ 
	unknown_0:Èd
	unknown_1:d 
	unknown_2:Èd
	unknown_3:d 
	unknown_4:Èd
	unknown_5:d 
	unknown_6:Èd
	unknown_7:d 
	unknown_8:Èd
	unknown_9:d!

unknown_10:Èd

unknown_11:d

unknown_12:
¼

unknown_13:	

unknown_14:	

unknown_15:
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220551o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


õ
C__inference_dense_1_layer_call_and_return_conditional_losses_220544

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ö
(__inference_PorkCNN_layer_call_fn_221041

inputs
unknown:
õÈ 
	unknown_0:Èd
	unknown_1:d 
	unknown_2:Èd
	unknown_3:d 
	unknown_4:Èd
	unknown_5:d 
	unknown_6:Èd
	unknown_7:d 
	unknown_8:Èd
	unknown_9:d!

unknown_10:Èd

unknown_11:d

unknown_12:
¼

unknown_13:	

unknown_14:	

unknown_15:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220551o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù	
b
C__inference_dropout_layer_call_and_return_conditional_losses_220618

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
a
C__inference_dropout_layer_call_and_return_conditional_losses_220531

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö

D__inference_conv1d_1_layer_call_and_return_conditional_losses_220404

inputsB
+conv1d_expanddims_1_readvariableop_resource:Èd-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¡
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èd®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
««
ß
!__inference__wrapped_model_220332
input_1=
)porkcnn_embedding_embedding_lookup_220213:
õÈQ
:porkcnn_conv1d_conv1d_expanddims_1_readvariableop_resource:Èd<
.porkcnn_conv1d_biasadd_readvariableop_resource:dS
<porkcnn_conv1d_1_conv1d_expanddims_1_readvariableop_resource:Èd>
0porkcnn_conv1d_1_biasadd_readvariableop_resource:dS
<porkcnn_conv1d_2_conv1d_expanddims_1_readvariableop_resource:Èd>
0porkcnn_conv1d_2_biasadd_readvariableop_resource:dS
<porkcnn_conv1d_3_conv1d_expanddims_1_readvariableop_resource:Èd>
0porkcnn_conv1d_3_biasadd_readvariableop_resource:dS
<porkcnn_conv1d_4_conv1d_expanddims_1_readvariableop_resource:Èd>
0porkcnn_conv1d_4_biasadd_readvariableop_resource:dS
<porkcnn_conv1d_5_conv1d_expanddims_1_readvariableop_resource:Èd>
0porkcnn_conv1d_5_biasadd_readvariableop_resource:d@
,porkcnn_dense_matmul_readvariableop_resource:
¼<
-porkcnn_dense_biasadd_readvariableop_resource:	A
.porkcnn_dense_1_matmul_readvariableop_resource:	=
/porkcnn_dense_1_biasadd_readvariableop_resource:
identity¢%PorkCNN/conv1d/BiasAdd/ReadVariableOp¢1PorkCNN/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢'PorkCNN/conv1d_1/BiasAdd/ReadVariableOp¢3PorkCNN/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢'PorkCNN/conv1d_2/BiasAdd/ReadVariableOp¢3PorkCNN/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp¢'PorkCNN/conv1d_3/BiasAdd/ReadVariableOp¢3PorkCNN/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢'PorkCNN/conv1d_4/BiasAdd/ReadVariableOp¢)PorkCNN/conv1d_4/BiasAdd_1/ReadVariableOp¢3PorkCNN/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢5PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp¢'PorkCNN/conv1d_5/BiasAdd/ReadVariableOp¢3PorkCNN/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢$PorkCNN/dense/BiasAdd/ReadVariableOp¢#PorkCNN/dense/MatMul/ReadVariableOp¢&PorkCNN/dense_1/BiasAdd/ReadVariableOp¢%PorkCNN/dense_1/MatMul/ReadVariableOp¢"PorkCNN/embedding/embedding_lookupò
"PorkCNN/embedding/embedding_lookupResourceGather)porkcnn_embedding_embedding_lookup_220213input_1*
Tindices0*<
_class2
0.loc:@PorkCNN/embedding/embedding_lookup/220213*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0Ú
+PorkCNN/embedding/embedding_lookup/IdentityIdentity+PorkCNN/embedding/embedding_lookup:output:0*
T0*<
_class2
0.loc:@PorkCNN/embedding/embedding_lookup/220213*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ§
-PorkCNN/embedding/embedding_lookup/Identity_1Identity4PorkCNN/embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈo
$PorkCNN/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÑ
 PorkCNN/conv1d/Conv1D/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:0-PorkCNN/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ±
1PorkCNN/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:porkcnn_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0h
&PorkCNN/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Î
"PorkCNN/conv1d/Conv1D/ExpandDims_1
ExpandDims9PorkCNN/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0/PorkCNN/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÈdÛ
PorkCNN/conv1d/Conv1DConv2D)PorkCNN/conv1d/Conv1D/ExpandDims:output:0+PorkCNN/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides

PorkCNN/conv1d/Conv1D/SqueezeSqueezePorkCNN/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
%PorkCNN/conv1d/BiasAdd/ReadVariableOpReadVariableOp.porkcnn_conv1d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¯
PorkCNN/conv1d/BiasAddBiasAdd&PorkCNN/conv1d/Conv1D/Squeeze:output:0-PorkCNN/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
PorkCNN/conv1d/ReluReluPorkCNN/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
2PorkCNN/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¹
 PorkCNN/global_max_pooling1d/MaxMax!PorkCNN/conv1d/Relu:activations:0;PorkCNN/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
&PorkCNN/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
"PorkCNN/conv1d_1/Conv1D/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:0/PorkCNN/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈµ
3PorkCNN/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<porkcnn_conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0j
(PorkCNN/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
$PorkCNN/conv1d_1/Conv1D/ExpandDims_1
ExpandDims;PorkCNN/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:01PorkCNN/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èdá
PorkCNN/conv1d_1/Conv1DConv2D+PorkCNN/conv1d_1/Conv1D/ExpandDims:output:0-PorkCNN/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides
£
PorkCNN/conv1d_1/Conv1D/SqueezeSqueeze PorkCNN/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'PorkCNN/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0porkcnn_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0µ
PorkCNN/conv1d_1/BiasAddBiasAdd(PorkCNN/conv1d_1/Conv1D/Squeeze:output:0/PorkCNN/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
PorkCNN/conv1d_1/ReluRelu!PorkCNN/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
4PorkCNN/global_max_pooling1d/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¿
"PorkCNN/global_max_pooling1d/Max_1Max#PorkCNN/conv1d_1/Relu:activations:0=PorkCNN/global_max_pooling1d/Max_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
&PorkCNN/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
"PorkCNN/conv1d_2/Conv1D/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:0/PorkCNN/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈµ
3PorkCNN/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<porkcnn_conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0j
(PorkCNN/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
$PorkCNN/conv1d_2/Conv1D/ExpandDims_1
ExpandDims;PorkCNN/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:01PorkCNN/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èdá
PorkCNN/conv1d_2/Conv1DConv2D+PorkCNN/conv1d_2/Conv1D/ExpandDims:output:0-PorkCNN/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides
£
PorkCNN/conv1d_2/Conv1D/SqueezeSqueeze PorkCNN/conv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'PorkCNN/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0porkcnn_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0µ
PorkCNN/conv1d_2/BiasAddBiasAdd(PorkCNN/conv1d_2/Conv1D/Squeeze:output:0/PorkCNN/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
PorkCNN/conv1d_2/ReluRelu!PorkCNN/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
4PorkCNN/global_max_pooling1d/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¿
"PorkCNN/global_max_pooling1d/Max_2Max#PorkCNN/conv1d_2/Relu:activations:0=PorkCNN/global_max_pooling1d/Max_2/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
&PorkCNN/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
"PorkCNN/conv1d_3/Conv1D/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:0/PorkCNN/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈµ
3PorkCNN/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<porkcnn_conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0j
(PorkCNN/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
$PorkCNN/conv1d_3/Conv1D/ExpandDims_1
ExpandDims;PorkCNN/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:01PorkCNN/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èdá
PorkCNN/conv1d_3/Conv1DConv2D+PorkCNN/conv1d_3/Conv1D/ExpandDims:output:0-PorkCNN/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides
£
PorkCNN/conv1d_3/Conv1D/SqueezeSqueeze PorkCNN/conv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'PorkCNN/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0porkcnn_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0µ
PorkCNN/conv1d_3/BiasAddBiasAdd(PorkCNN/conv1d_3/Conv1D/Squeeze:output:0/PorkCNN/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
PorkCNN/conv1d_3/ReluRelu!PorkCNN/conv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
4PorkCNN/global_max_pooling1d/Max_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¿
"PorkCNN/global_max_pooling1d/Max_3Max#PorkCNN/conv1d_3/Relu:activations:0=PorkCNN/global_max_pooling1d/Max_3/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
&PorkCNN/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
"PorkCNN/conv1d_4/Conv1D/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:0/PorkCNN/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈµ
3PorkCNN/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<porkcnn_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0j
(PorkCNN/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
$PorkCNN/conv1d_4/Conv1D/ExpandDims_1
ExpandDims;PorkCNN/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:01PorkCNN/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èdá
PorkCNN/conv1d_4/Conv1DConv2D+PorkCNN/conv1d_4/Conv1D/ExpandDims:output:0-PorkCNN/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides
£
PorkCNN/conv1d_4/Conv1D/SqueezeSqueeze PorkCNN/conv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'PorkCNN/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp0porkcnn_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0µ
PorkCNN/conv1d_4/BiasAddBiasAdd(PorkCNN/conv1d_4/Conv1D/Squeeze:output:0/PorkCNN/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
PorkCNN/conv1d_4/ReluRelu!PorkCNN/conv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
4PorkCNN/global_max_pooling1d/Max_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¿
"PorkCNN/global_max_pooling1d/Max_4Max#PorkCNN/conv1d_4/Relu:activations:0=PorkCNN/global_max_pooling1d/Max_4/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
(PorkCNN/conv1d_4/Conv1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÙ
$PorkCNN/conv1d_4/Conv1D_1/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:01PorkCNN/conv1d_4/Conv1D_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ·
5PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOpReadVariableOp<porkcnn_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0l
*PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ú
&PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1
ExpandDims=PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp:value:03PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èdç
PorkCNN/conv1d_4/Conv1D_1Conv2D-PorkCNN/conv1d_4/Conv1D_1/ExpandDims:output:0/PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides
§
!PorkCNN/conv1d_4/Conv1D_1/SqueezeSqueeze"PorkCNN/conv1d_4/Conv1D_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
)PorkCNN/conv1d_4/BiasAdd_1/ReadVariableOpReadVariableOp0porkcnn_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0»
PorkCNN/conv1d_4/BiasAdd_1BiasAdd*PorkCNN/conv1d_4/Conv1D_1/Squeeze:output:01PorkCNN/conv1d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
PorkCNN/conv1d_4/Relu_1Relu#PorkCNN/conv1d_4/BiasAdd_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
4PorkCNN/global_max_pooling1d/Max_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Á
"PorkCNN/global_max_pooling1d/Max_5Max%PorkCNN/conv1d_4/Relu_1:activations:0=PorkCNN/global_max_pooling1d/Max_5/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
&PorkCNN/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÕ
"PorkCNN/conv1d_5/Conv1D/ExpandDims
ExpandDims6PorkCNN/embedding/embedding_lookup/Identity_1:output:0/PorkCNN/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈµ
3PorkCNN/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<porkcnn_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:Èd*
dtype0j
(PorkCNN/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
$PorkCNN/conv1d_5/Conv1D/ExpandDims_1
ExpandDims;PorkCNN/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:01PorkCNN/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Èdá
PorkCNN/conv1d_5/Conv1DConv2D+PorkCNN/conv1d_5/Conv1D/ExpandDims:output:0-PorkCNN/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingVALID*
strides
£
PorkCNN/conv1d_5/Conv1D/SqueezeSqueeze PorkCNN/conv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'PorkCNN/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp0porkcnn_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0µ
PorkCNN/conv1d_5/BiasAddBiasAdd(PorkCNN/conv1d_5/Conv1D/Squeeze:output:0/PorkCNN/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
PorkCNN/conv1d_5/ReluRelu!PorkCNN/conv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
4PorkCNN/global_max_pooling1d/Max_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¿
"PorkCNN/global_max_pooling1d/Max_6Max#PorkCNN/conv1d_5/Relu:activations:0=PorkCNN/global_max_pooling1d/Max_6/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
PorkCNN/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ­
PorkCNN/concatConcatV2)PorkCNN/global_max_pooling1d/Max:output:0+PorkCNN/global_max_pooling1d/Max_1:output:0+PorkCNN/global_max_pooling1d/Max_2:output:0+PorkCNN/global_max_pooling1d/Max_3:output:0+PorkCNN/global_max_pooling1d/Max_4:output:0+PorkCNN/global_max_pooling1d/Max_5:output:0+PorkCNN/global_max_pooling1d/Max_6:output:0PorkCNN/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#PorkCNN/dense/MatMul/ReadVariableOpReadVariableOp,porkcnn_dense_matmul_readvariableop_resource* 
_output_shapes
:
¼*
dtype0
PorkCNN/dense/MatMulMatMulPorkCNN/concat:output:0+PorkCNN/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$PorkCNN/dense/BiasAdd/ReadVariableOpReadVariableOp-porkcnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
PorkCNN/dense/BiasAddBiasAddPorkCNN/dense/MatMul:product:0,PorkCNN/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
PorkCNN/dense/ReluReluPorkCNN/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
PorkCNN/dropout/IdentityIdentity PorkCNN/dense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%PorkCNN/dense_1/MatMul/ReadVariableOpReadVariableOp.porkcnn_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¤
PorkCNN/dense_1/MatMulMatMul!PorkCNN/dropout/Identity:output:0-PorkCNN/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&PorkCNN/dense_1/BiasAdd/ReadVariableOpReadVariableOp/porkcnn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
PorkCNN/dense_1/BiasAddBiasAdd PorkCNN/dense_1/MatMul:product:0.PorkCNN/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
PorkCNN/dense_1/SigmoidSigmoid PorkCNN/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityPorkCNN/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp&^PorkCNN/conv1d/BiasAdd/ReadVariableOp2^PorkCNN/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^PorkCNN/conv1d_1/BiasAdd/ReadVariableOp4^PorkCNN/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp(^PorkCNN/conv1d_2/BiasAdd/ReadVariableOp4^PorkCNN/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp(^PorkCNN/conv1d_3/BiasAdd/ReadVariableOp4^PorkCNN/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp(^PorkCNN/conv1d_4/BiasAdd/ReadVariableOp*^PorkCNN/conv1d_4/BiasAdd_1/ReadVariableOp4^PorkCNN/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp6^PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp(^PorkCNN/conv1d_5/BiasAdd/ReadVariableOp4^PorkCNN/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp%^PorkCNN/dense/BiasAdd/ReadVariableOp$^PorkCNN/dense/MatMul/ReadVariableOp'^PorkCNN/dense_1/BiasAdd/ReadVariableOp&^PorkCNN/dense_1/MatMul/ReadVariableOp#^PorkCNN/embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2N
%PorkCNN/conv1d/BiasAdd/ReadVariableOp%PorkCNN/conv1d/BiasAdd/ReadVariableOp2f
1PorkCNN/conv1d/Conv1D/ExpandDims_1/ReadVariableOp1PorkCNN/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'PorkCNN/conv1d_1/BiasAdd/ReadVariableOp'PorkCNN/conv1d_1/BiasAdd/ReadVariableOp2j
3PorkCNN/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp3PorkCNN/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2R
'PorkCNN/conv1d_2/BiasAdd/ReadVariableOp'PorkCNN/conv1d_2/BiasAdd/ReadVariableOp2j
3PorkCNN/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp3PorkCNN/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2R
'PorkCNN/conv1d_3/BiasAdd/ReadVariableOp'PorkCNN/conv1d_3/BiasAdd/ReadVariableOp2j
3PorkCNN/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp3PorkCNN/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2R
'PorkCNN/conv1d_4/BiasAdd/ReadVariableOp'PorkCNN/conv1d_4/BiasAdd/ReadVariableOp2V
)PorkCNN/conv1d_4/BiasAdd_1/ReadVariableOp)PorkCNN/conv1d_4/BiasAdd_1/ReadVariableOp2j
3PorkCNN/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp3PorkCNN/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2n
5PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp5PorkCNN/conv1d_4/Conv1D_1/ExpandDims_1/ReadVariableOp2R
'PorkCNN/conv1d_5/BiasAdd/ReadVariableOp'PorkCNN/conv1d_5/BiasAdd/ReadVariableOp2j
3PorkCNN/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp3PorkCNN/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2L
$PorkCNN/dense/BiasAdd/ReadVariableOp$PorkCNN/dense/BiasAdd/ReadVariableOp2J
#PorkCNN/dense/MatMul/ReadVariableOp#PorkCNN/dense/MatMul/ReadVariableOp2P
&PorkCNN/dense_1/BiasAdd/ReadVariableOp&PorkCNN/dense_1/BiasAdd/ReadVariableOp2N
%PorkCNN/dense_1/MatMul/ReadVariableOp%PorkCNN/dense_1/MatMul/ReadVariableOp2H
"PorkCNN/embedding/embedding_lookup"PorkCNN/embedding/embedding_lookup:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ù

'__inference_conv1d_layer_call_fn_221397

inputs
unknown:Èd
	unknown_0:d
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_220381t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þÁ
ô
	embedding

bigram
bigram2
trigram
trigram2
fourgram
fivegram
pool
	dense_1

dropout

last_dense
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
µ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
»

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
»

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
»

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
§
iiter

jbeta_1

kbeta_2
	ldecay
mlearning_ratem¶m·m¸$m¹%mº,m»-m¼4m½5m¾<m¿=mÀDmÁEmÂRmÃSmÄamÅbmÆvÇvÈvÉ$vÊ%vË,vÌ-vÍ4vÎ5vÏ<vÐ=vÑDvÒEvÓRvÔSvÕavÖbv×"
	optimizer

0
1
2
$3
%4
,5
-6
47
58
<9
=10
D11
E12
R13
S14
a15
b16"
trackable_list_wrapper

0
1
2
$3
%4
,5
-6
47
58
<9
=10
D11
E12
R13
S14
a15
b16"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Þ2Û
(__inference_PorkCNN_layer_call_fn_220588
(__inference_PorkCNN_layer_call_fn_221041
(__inference_PorkCNN_layer_call_fn_221080
(__inference_PorkCNN_layer_call_fn_220876°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
C__inference_PorkCNN_layer_call_and_return_conditional_losses_221202
C__inference_PorkCNN_layer_call_and_return_conditional_losses_221331
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220936
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220996°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
!__inference__wrapped_model_220332input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
sserving_default"
signature_map
(:&
õÈ2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_embedding_layer_call_fn_221379¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_embedding_layer_call_and_return_conditional_losses_221388¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
$:"Èd2conv1d/kernel
:d2conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_conv1d_layer_call_fn_221397¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv1d_layer_call_and_return_conditional_losses_221413¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$Èd2conv1d_1/kernel
:d2conv1d_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv1d_1_layer_call_fn_221422¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_1_layer_call_and_return_conditional_losses_221438¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$Èd2conv1d_2/kernel
:d2conv1d_2/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv1d_2_layer_call_fn_221447¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_2_layer_call_and_return_conditional_losses_221463¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$Èd2conv1d_3/kernel
:d2conv1d_3/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv1d_3_layer_call_fn_221472¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_3_layer_call_and_return_conditional_losses_221488¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$Èd2conv1d_4/kernel
:d2conv1d_4/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv1d_4_layer_call_fn_221497¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_4_layer_call_and_return_conditional_losses_221513¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$Èd2conv1d_5/kernel
:d2conv1d_5/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv1d_5_layer_call_fn_221522¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_5_layer_call_and_return_conditional_losses_221538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_global_max_pooling1d_layer_call_fn_221543¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_221549¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :
¼2dense/kernel
:2
dense/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_layer_call_fn_221558¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_221569¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_dropout_layer_call_fn_221574
(__inference_dropout_layer_call_fn_221579´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_layer_call_and_return_conditional_losses_221584
C__inference_dropout_layer_call_and_return_conditional_losses_221596´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:	2dense_1/kernel
:2dense_1/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_1_layer_call_fn_221605¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_221616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
$__inference_signature_wrapper_221372input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

­total

®count
¯	variables
°	keras_api"
_tf_keras_metric
c

±total

²count
³
_fn_kwargs
´	variables
µ	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
­0
®1"
trackable_list_wrapper
.
¯	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
±0
²1"
trackable_list_wrapper
.
´	variables"
_generic_user_object
-:+
õÈ2Adam/embedding/embeddings/m
):'Èd2Adam/conv1d/kernel/m
:d2Adam/conv1d/bias/m
+:)Èd2Adam/conv1d_1/kernel/m
 :d2Adam/conv1d_1/bias/m
+:)Èd2Adam/conv1d_2/kernel/m
 :d2Adam/conv1d_2/bias/m
+:)Èd2Adam/conv1d_3/kernel/m
 :d2Adam/conv1d_3/bias/m
+:)Èd2Adam/conv1d_4/kernel/m
 :d2Adam/conv1d_4/bias/m
+:)Èd2Adam/conv1d_5/kernel/m
 :d2Adam/conv1d_5/bias/m
%:#
¼2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
-:+
õÈ2Adam/embedding/embeddings/v
):'Èd2Adam/conv1d/kernel/v
:d2Adam/conv1d/bias/v
+:)Èd2Adam/conv1d_1/kernel/v
 :d2Adam/conv1d_1/bias/v
+:)Èd2Adam/conv1d_2/kernel/v
 :d2Adam/conv1d_2/bias/v
+:)Èd2Adam/conv1d_3/kernel/v
 :d2Adam/conv1d_3/bias/v
+:)Èd2Adam/conv1d_4/kernel/v
 :d2Adam/conv1d_4/bias/v
+:)Èd2Adam/conv1d_5/kernel/v
 :d2Adam/conv1d_5/bias/v
%:#
¼2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v¸
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220936q$%,-45<=DERSab5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
C__inference_PorkCNN_layer_call_and_return_conditional_losses_220996q$%,-45<=DERSab5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_PorkCNN_layer_call_and_return_conditional_losses_221202p$%,-45<=DERSab4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_PorkCNN_layer_call_and_return_conditional_losses_221331p$%,-45<=DERSab4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_PorkCNN_layer_call_fn_220588d$%,-45<=DERSab5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_PorkCNN_layer_call_fn_220876d$%,-45<=DERSab5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_PorkCNN_layer_call_fn_221041c$%,-45<=DERSab4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_PorkCNN_layer_call_fn_221080c$%,-45<=DERSab4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ 
!__inference__wrapped_model_220332{$%,-45<=DERSab1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ¯
D__inference_conv1d_1_layer_call_and_return_conditional_losses_221438g$%5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿd
 
)__inference_conv1d_1_layer_call_fn_221422Z$%5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿd¯
D__inference_conv1d_2_layer_call_and_return_conditional_losses_221463g,-5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿd
 
)__inference_conv1d_2_layer_call_fn_221447Z,-5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿd¯
D__inference_conv1d_3_layer_call_and_return_conditional_losses_221488g455¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿd
 
)__inference_conv1d_3_layer_call_fn_221472Z455¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿd¯
D__inference_conv1d_4_layer_call_and_return_conditional_losses_221513g<=5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿd
 
)__inference_conv1d_4_layer_call_fn_221497Z<=5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿd¯
D__inference_conv1d_5_layer_call_and_return_conditional_losses_221538gDE5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿd
 
)__inference_conv1d_5_layer_call_fn_221522ZDE5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿd­
B__inference_conv1d_layer_call_and_return_conditional_losses_221413g5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿd
 
'__inference_conv1d_layer_call_fn_221397Z5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿd¤
C__inference_dense_1_layer_call_and_return_conditional_losses_221616]ab0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_1_layer_call_fn_221605Pab0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_dense_layer_call_and_return_conditional_losses_221569^RS0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¼
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dense_layer_call_fn_221558QRS0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¼
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dropout_layer_call_and_return_conditional_losses_221584^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¥
C__inference_dropout_layer_call_and_return_conditional_losses_221596^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dropout_layer_call_fn_221574Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
(__inference_dropout_layer_call_fn_221579Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ«
E__inference_embedding_layer_call_and_return_conditional_losses_221388b0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿÈ
 
*__inference_embedding_layer_call_fn_221379U0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÈË
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_221549wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 £
5__inference_global_max_pooling1d_layer_call_fn_221543jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
$__inference_signature_wrapper_221372$%,-45<=DERSab<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ