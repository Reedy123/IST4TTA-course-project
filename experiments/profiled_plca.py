"""从官方源文件生成仅加计时作用域的内存副本，不改写官方文件。"""
import ast
import copy
from pathlib import Path
from .profiling_runtime import profiler, timed_cg


def scope(name, statements, gpu=True, force=False):
    """添加只读计时作用域，原语句和顺序保持不变。"""
    context=ast.Call(func=ast.Attribute(value=ast.Name(id='prof',ctx=ast.Load()),attr='stage',ctx=ast.Load()),
                     args=[ast.Constant(name)],keywords=[ast.keyword(arg='gpu',value=ast.Constant(gpu)),ast.keyword(arg='force',value=ast.Constant(force))])
    return ast.With(items=[ast.withitem(context_expr=context,optional_vars=None)],body=statements,type_comment=None)


def group(body, ranges):
    """只包围指定原始行区间，保持分支和变量作用域。"""
    result=[];i=0
    while i<len(body):
        node=body[i];match=next(((a,b,name) for a,b,name in ranges if a<=node.lineno<=b),None)
        if match:
            a,b,name=match;chunk=[]
            while i<len(body) and a<=body[i].lineno<=b:chunk.append(body[i]);i+=1
            result.append(scope(name,chunk))
        else:result.append(node);i+=1
    return result


class Instrument(ast.NodeTransformer):
    """仅替换cg为透明计时包装，并添加阶段边界。"""
    def visit_Call(self,node):
        """不改变SciPy的任何实参。"""
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='cg':node.func.id='timed_cg'
        return node

    def visit_FunctionDef(self,node):
        """官方固定版本行号用于精确界定原始计算块。"""
        self.generic_visit(node)
        if node.name=='PLCA':
            node.body=group(node.body,[(105,110,'plca_setup'),(113,128,'faiss'),(130,154,'graph'),(156,165,'graph_cpu_bridge'),(168,183,'propagation_other')])
            node.body=[scope('plca_total',node.body,gpu=False,force=True)]
        if node.name=='soft_labeling':
            for item in ast.walk(node):
                if isinstance(item,ast.For) and item.lineno==56:
                    item.body=group(item.body,[(57,57,'input_transfer'),(58,58,'feature_forward'),(60,74,'prediction_collection')])
        return node


class Strip(ast.NodeTransformer):
    """还原作用域并核验算法AST没有变化。"""
    def visit_With(self,node):
        """去掉计时with，保留所有原语句。"""
        self.generic_visit(node)
        c=node.items[0].context_expr
        if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='prof':return node.body
        return node

    def visit_Call(self,node):
        """还原cg符号以作严格语义结构对照。"""
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='timed_cg':node.func.id='cg'
        return node


def instrument_tree(source):
    """构建并核验只加时序观察的函数副本。"""
    original=ast.parse(source);tree=Instrument().visit(copy.deepcopy(original));ast.fix_missing_locations(tree)
    assert ast.dump(Strip().visit(copy.deepcopy(tree)))==ast.dump(original),'计时插桩改变了算法AST'
    return tree


def load_profiled(path):
    """在utils包上下文编译内存副本，复用同一个全局记录器。"""
    path=Path(path);tree=instrument_tree(path.read_text())
    namespace={'__name__':'utils.profiled_pseudo_labeling','__package__':'utils','__file__':str(path),'prof':profiler,'timed_cg':timed_cg}
    exec(compile(tree,str(path)+'[profiled]','exec'),namespace)
    return namespace['robust_PLCA']
