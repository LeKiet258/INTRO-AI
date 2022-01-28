using CSV
using DataFrames
using Random
using StatsBase

#-------------------------------------------CLASS-------------------------------------------
mutable struct Node
    feature # feature to make the split and branches
    children::Vector{Any} 
    value_cutoff # cutoff of feature
    is_leaf

    # constructor
    Node(fea, chi = [], val = 0, lea = false) = new(fea, chi, val, lea)
end

#-------------------------------------------UTILS-------------------------------------------
function splitdf(df, pct)
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct

    train_set, test_set = view(df, sel, :), view(df, .!sel, :)
    X_train, y_train = train_set[:, 2:end-1], train_set."Species"
    X_test, y_test = test_set[:, 2:end-1], test_set."Species"

    return X_train, y_train, X_test, y_test
end

function get_entropy(y)
    probs = values(countmap(y)) ./ length(y)
    entropy = sum(-probs .* log2.(probs))
    return entropy
end

function find_cutoff(X, y, col)
    min_entropy = 10
    correct_cutoff = 0 # index

    # scan all rows
    for cutoff in unique(X[!, col])
        # calc entropy(D <= cutoff)
        smaller = y[X[!, col] .<= cutoff, :]
        entropy_smaller = get_entropy(smaller)
        
        # calc entropy(D > cutoff)
        bigger = y[X[!, col] .> cutoff, :]
        entropy_bigger = get_entropy(bigger)
    
        # entropy of cutoff
        entropy_cutoff = length(smaller)/nrow(X)*entropy_smaller + length(bigger)/nrow(X)*entropy_bigger

        if min_entropy > entropy_cutoff
            min_entropy = entropy_cutoff
            correct_cutoff = cutoff
        end 
    end
    return correct_cutoff, min_entropy
end

# ID3 
function construct_tree(X, y, tree)
    features = names(X) # remaining cols
    
    # if y_train is purity
    if length(unique(y)) == 1
        tree.is_leaf = true 
        tree.feature = y[1]
        return tree 
    end

    # if no more data in X_train 
    if nrow(X) == 0
        tree.is_leaf = true
        tree.feature = sort(collect(countmap(y)))[1][1] # more than 2 labels? -> assign the most frequent label
        return tree
    end

    # find cutoff & feature with min entropy
    cutoffs = [find_cutoff(X, y, col) for col in features] # cutoffs of all features: list of tuples
    max_gain_ind = findmin([x[2] for x in cutoffs])[2] # index of feature with min entropy

    # init root 
    tree.feature = features[max_gain_ind]
    tree.children = [Node(nothing), Node(nothing)] # root w/ two empty children
    tree.value_cutoff = cutoffs[max_gain_ind][1] 

    # construct left subtree 
    mask = X[!, tree.feature] .<= tree.value_cutoff # bool vector
    X_low, y_low = X[mask, Not(tree.feature)], y[mask] # drop tree.feature for the next iter
    construct_tree(X_low, y_low, tree.children[1])

    # construct right subtree
    mask = .!(mask) # negation(≤) = (>) -> bool vector
    X_high, y_high = X[mask, Not(tree.feature)], y[mask] # drop tree.feature for the next iter
    construct_tree(X_high, y_high, tree.children[2])

    return tree
end

# predict for 1 row
function predictor(row, tree)
    if tree.is_leaf == true
        return tree.feature
    end
    if row[tree.feature] <= tree.value_cutoff
        return predictor(row, tree.children[1])
    end
    return predictor(row, tree.children[2])
end

# predict for test set
function predict_test_set(X_test, tree)
    y_pred = []
    for row in eachrow(X_test)
        push!(y_pred, predictor(row, tree))
    end
    return y_pred
end

#-------------------------------------------MAIN-------------------------------------------
function main()
    # read dataset & split into train_set and test_set
    dataset = DataFrame(CSV.File("iris.csv"));
    X_train, y_train, X_test, y_test = splitdf(dataset, 2/3);

    # construct tree
    tree = Node(nothing)
    construct_tree(X_train, y_train, tree)

    # test_set prediction
    correct_pred = 0
    y_pred = predict_test_set(X_test, tree);
    for i=1:length(y_pred)
        if y_pred[i] == y_test[i]
            correct_pred += 1
        end
    end
    println("Accuracy on test set: ", correct_pred/length(y_test))
end

main()