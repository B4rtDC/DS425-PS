classdef Wallet
    properties
        cash {mustBeNumeric}
        crypto {mustBeNumeric} % can be extende to hold multiple cryptocurrencies
    end
    methods
        function obj = Wallet(cash, crypto)
            if nargin == 0
                obj.cash = 0;
                obj.crypto = 0;
            elseif nargin == 2
                obj.cash = cash;
                obj.crypto = crypto;
            end
        end
        
        function obj = buy(obj, unitprice, NameValueArgs)
            % BUY use available cash to buy more crypto
            %
            % buy(obj, unitprice) buy as much as possible given the current
            % unitprice of the cryptocurrency
            %
            % buy(obj, unitprice, 'volume', x) buy x percent of the
            % volume of the cryptocurrency (default = 1)
            %
            % buy(obj, unitprice, 'tradingcost',x) take into account a
            % tradingcost for the exchange (default = 0)
            arguments
                obj
                unitprice
                NameValueArgs.tradingcost = 0
                NameValueArgs.volume = 1
            end
            obj.crypto = obj.crypto + obj.cash * NameValueArgs.volume / unitprice *(1-NameValueArgs.tradingcost);
            obj.cash = obj.cash*(1- NameValueArgs.volume);
        end
        
        function obj = sell(obj, unitprice, NameValueArgs)
            % SELL use available crypto to obtain more cash
            %
            % sell(obj, unitprice) sell as much as possible given the current
            % unitprice of the cryptocurrency
            %
            % sell(obj, unitprice, 'volume', x) sell x percent of the
            % volume of the cryptocurrency (default = 1)
            %
            % sell(obj, unitprice, 'tradingcost',x) take into account a
            % tradingcost for the exchange (default = 0)
            arguments
                obj
                unitprice
                NameValueArgs.tradingcost = 0
                NameValueArgs.volume = 1
            end
            obj.cash = obj.cash + obj.crypto * NameValueArgs.volume * unitprice*(1-NameValueArgs.tradingcost);
            obj.crypto = obj.crypto * (1-NameValueArgs.volume);
        end
        
        function v = value(obj, unitprice)
            % VALUE return the total wallet value given the current
            % unitprice of the cryptocurrency
            v = obj.cash + obj.crypto * unitprice;
        end
    end
end
