namespace FunctionalProgramming.Web.Events;

public abstract record Event(Guid EntityId, DateTime Timestamp);
