# 内联函数
- 有些函数被频繁调用，不断有函数进栈出栈
- 调用内联函数的时候，会把内联函数直接替换为函数的实现
- inline修饰符必须与函数定义/实现放在一起，和函数声明/签名放在一起没有用
- 所以说inline是一种用于实现的关键字
- 内联函数中不能包含复杂的结构控制语句如while，switch