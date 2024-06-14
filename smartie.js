function Value(data, children, op) {
    if (!(this instanceof Value))
        return new Value(data, children, op);

    if (typeof data != 'number')
        throw new Error("Value(d): underlying type of d must be a number.");

    this._data     = data;
    this._children = children || [];
    this._op       = op;
    this._label    = null;
    this._grad     = 0;
}

const Operation = {
    PLUS: "_Add",
    MUL: "_Mul"
};

Value.prototype = {
    add(x) {
        return Value(this._data + x._data, [this, x], '_Add');
    },

    mul(x) {
        return Value(this._data * x._data, [this, x], '_Mul');
    },

    relu() {
        return Value(relu(this._data), [this], '_ReLu');
    },

    tanh() {
        return Value(tanh(this._data), [this], '_Tanh');
    },
    
    setLabel(l) {
        this._label = l;
        return this;
    },

    _backpropagate(currNode, grad) {
        const localGrad = currNode.gradient();
        if (!localGrad)
            throw new Error("couldn't propagate back: forward node doenst have a valid gradient object.");
        
        for (let i = 0; i < this._children.length; ++i) {
            const prev = this._children[i];

            if (currNode._op == Operation.MUL)
                prev._grad += localGrad(i) * grad;
            else prev._grad += localGrad() * grad;

            
            if (prev._children.length > 0)
                prev._backpropagate(prev, prev._grad);
        }
    
    },
    
    gradient() {
        if (!this._op || !(this._op in gradTable))
            return null;

        return gradTable[this._op](this);
    },

    backPropag() {
        this._grad = 1;
        
        this._backpropagate(this, this._grad);
    },

    data() {
        return this._data;
    }
};

function Tensor(data, shape) {
    if (!(this instanceof Tensor))
        return new Tensor(data, shape);

    this._data  = data;
    this._shape = shape;

    if (!this._data)
        this._data = [];
    
    this._init();
}

Tensor.prototype = {
    _init() {
        for (let i = 0; i < this._data.length; ++i) {
            if (!(this._data[i] instanceof Value))
                this._data[i] = Value(this._data[i]);
        }
    }
}

function Shape(r, c) {
    if (!(this instanceof Shape))
        return new Shape(r, c);

    this._rows = r;
    this._cols = c;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function relu(x) {
    return Math.max(0, x);
}

function tanh(x) {
    return (Math.exp(2*x) - 1) / (Math.exp(2 * x) + 1);
}

function tanhGrad(x) {
    return 1 - Math.pow(tanh(x), 2);
}


function reluGrad(x) {
    return x <= 0 ? 0 : 1;
}

function sigmoidGrad(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}


class Matrix {
    constructor(r, c) {
        this.rows  = r;
        this.cols  = c;
        this._data = [];

        this._init();
    }

    _init() {
        for (let i = 0; i < this.rows * this.cols; ++i)
            this._data[i] = Value(0);
    }

    set(i, j, val) {
        if (i >= this.rows || j >= this.cols || i < 0 || j < 0)
            return;
        
        this._data[i * this.cols + j] = val;
    }
}

function RandomMatrix(r, c) {
    const m = new Matrix(r, c);

    for (let i = 0; i < r; ++i) {
        for (let j = 0; j < c; ++j) {
            m.set(i, j, Value(Math.random()*2 - 1));
        }
    }

    return m;
}

function nn(x) {
}


const gradTable = {
    "_Add": function(node) {
        return function() {
            return 1;
        }
    },

    "_Mul": function(node) {
        return function(operand) {
            if (operand > 1)
                return null;

            return node._children[1 - operand].data();
        }
    },

    "_ReLu": function(node) {
        return function() {
            return reluGrad(node._children[0].data());
        }
    },
}

